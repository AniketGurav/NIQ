import pandas as pd
import numpy as np
import spacy
import torch
import torch.nn as nn
import torch.optim as optim
import time
from sklearn.preprocessing import LabelEncoder, StandardScaler
import nltk
from nltk.tokenize import word_tokenize
from collections import Counter
import os
import re
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn import metrics
from tagging import compute_tag_counts

class NutritionalEntityClassifier:
    """
    Predicts nutritional and categorical attributes like energy, proteins, NutriScore, and food groups.

    Pseudo Logic:
        - Load 10,000 records, clean numerical targets (remove outliers), filter valid text
        - Tokenize text, encode brands/countries, normalize features/targets
        - Build vocab once, add tag counts, prepare inputs
        - Train BiLSTM with gradient clipping
        - Evaluate: RMSE for numerical, F1-score for categorical
    """
    
    def __init__(self, data_path='./cleaned_openfoodfacts.csv', 
                 glove_path='/cluster/datastore/aniketag/allData/NIQ/glove.6B.300d.txt',
                 target_columns=['energy_100g', 'fat_100g', 'proteins_100g', 'carbohydrates_100g', 
                                 'nutriscore_grade', 'pnns_groups_1']):
        """Set up paths, device, targets, and initialize NLTK/spaCy."""
        self.data_path = data_path
        self.glove_path = glove_path
        self.target_columns = target_columns
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        torch.cuda.empty_cache()
        self.spacy_en = spacy.load('en_core_web_trf')
        nltk.data.path.append("/cluster/datastore/aniketag/allData/nltk_data")
        if os.path.exists("/cluster/datastore/aniketag/allData/nltk_data/tokenizers/punkt"):
            nltk.download('punkt', download_dir="/cluster/datastore/aniketag/allData/nltk_data")
        self.numerical_cols = []
        self.categorical_cols = []
        self.label_vocabs = {}
        self.feature_cols = ['additives_n', 'serving_quantity']
        self.feature_scaler = StandardScaler()
        self.target_scaler = StandardScaler()

    def analyze_columns(self, data):
        """Identify numerical and categorical target/feature columns."""
        for col in self.target_columns:
            if col in data.columns:
                if data[col].dtype in ['float64', 'int64']:
                    self.numerical_cols.append(col)
                else:
                    self.categorical_cols.append(col)
            else:
                print(f"Warning: {col} not found in dataset.")
        print("Numerical targets:", self.numerical_cols)
        print("Categorical targets:", self.categorical_cols)

    def load_data(self):
        """Load 10,000 records, clean and normalize data."""
        df = pd.read_csv(self.data_path, nrows=10000)
        # Combine text fields
        df['text'] = df[['product_name', 'ingredients_text', 'categories']].fillna('').agg(' '.join, axis=1)
        # Tokenize and filter rows with non-empty text
        df['tokens'] = df['text'].apply(self.tokenize)
        df = df[df['tokens'].apply(len) > 0]
        # Clean numerical targets: remove outliers, impute, normalize
        for col in self.numerical_cols:
            if col in df.columns:
                # Convert to numeric, coerce errors to NaN
                df[col] = pd.to_numeric(df[col], errors='coerce')
                # Clip outliers to ±3 std, remove negatives
                mean, std = df[col].mean(), df[col].std()
                df = df[df[col].notna() & (df[col] >= 0) & (df[col] <= mean + 3 * std) & (df[col] >= mean - 3 * std)]
                # Impute NaNs with median
                df[col] = df[col].fillna(df[col].median())
                if df[col].isna().any():
                    raise ValueError(f"NaN values in {col} after imputation")
        # Filter rows with non-missing numerical targets
        critical_cols = [col for col in self.numerical_cols if col in df.columns]
        df = df.dropna(subset=critical_cols)
        # Normalize numerical targets
        if self.numerical_cols:
            df[self.numerical_cols] = self.target_scaler.fit_transform(df[self.numerical_cols])
        # Impute and normalize numerical features
        for col in self.feature_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                df[col] = df[col].fillna(df[col].median())
                if df[col].isna().any():
                    raise ValueError(f"NaN values in {col} after imputation")
        if self.feature_cols:
            df[self.feature_cols] = self.feature_scaler.fit_transform(df[self.feature_cols])
        # Handle categorical targets
        for col in self.categorical_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).replace(['nan', 'NaN', ''], 'unknown')
        # Encode categorical features
        for col in ['brands', 'countries_tags']:
            le = LabelEncoder()
            df[f'{col}_id'] = le.fit_transform(df[col].fillna('unknown'))
        print("Data shape after filtering:", df.shape)
        return df

    def tokenize(self, text):
        """Tokenize text into clean, lowercase words."""
        try:
            tokens = re.findall(r"\b\w+\b", str(text).lower())
            return tokens if tokens else ['unknown']
        except Exception:
            return ['unknown']

    def build_vocab(self, tokens):
        """Build vocab with pad/unk tokens."""
        counter = Counter(token for token_list in tokens for token in token_list)
        vocab = {word: idx + 2 for idx, (word, _) in enumerate(counter.most_common())}
        vocab['<pad>'] = 0
        vocab['<unk>'] = 1
        return vocab

    def build_label_vocab(self, labels):
        """Build vocab for categorical labels."""
        valid_labels = [str(label) for label in labels if str(label) not in ['nan', 'NaN', '']]
        unique_labels = sorted(set(valid_labels))
        if 'unknown' not in unique_labels:
            unique_labels.append('unknown')
        return {label: idx for idx, label in enumerate(unique_labels)}

    def tokens_to_ids(self, tokens, vocab):
        """Convert tokens to numeric IDs, ensuring valid indices."""
        ids = [vocab.get(token, vocab['<unk>']) for token in tokens]
        return [idx for idx in ids if idx < len(vocab)]

    def labels_to_ids(self, labels, label_vocab):
        """Convert labels to numeric IDs."""
        label_str = str(labels) if str(labels) not in ['nan', 'NaN', ''] else 'unknown'
        return label_vocab[label_str]

    def load_glove_embeddings(self, vocab):
        """Load GloVe vectors, create embedding matrix."""
        glove = {}
        embedding_dim = 300
        print("Loading GloVe vectors...", self.glove_path)
        with open(self.glove_path, 'r', encoding='utf-8') as f:
            for line in f:
                values = line.strip().split()
                word = values[0]
                vec = np.array(values[1:], dtype='float32')
                glove[word] = vec
        print("GloVe loaded successfully.")
        embedding_matrix = np.zeros((len(vocab), embedding_dim))
        print("Creating embedding matrix...", len(vocab), embedding_dim)
        for word, idx in vocab.items():
            embedding_matrix[idx] = glove.get(word, np.random.normal(scale=0.6, size=(embedding_dim,)))
        return embedding_matrix

    class NutritionDataset(Dataset):
        """Dataset for token IDs, features, and targets."""
        def __init__(self, input_ids, features, targets, numerical_cols):
            self.inputs = input_ids
            self.features = features
            self.targets = targets
            self.numerical_cols = numerical_cols

        def __getitem__(self, idx):
            return (
                torch.tensor(self.inputs[idx], dtype=torch.long),
                torch.tensor(self.features[idx], dtype=torch.float),
                {key: torch.tensor(val[idx], dtype=torch.float if key in self.numerical_cols else torch.long)
                 for key, val in self.targets.items()}
            )

        def __len__(self):
            return len(self.inputs)

    def collate_fn(self, batch):
        """Pad sequences and format batch."""
        inputs, features, targets = zip(*batch)
        padded_inputs = pad_sequence(inputs, batch_first=True, padding_value=0)
        target_batch = {key: [] for key in targets[0]}
        for t in targets:
            for key, val in t.items():
                target_batch[key].append(val)
        return (
            padded_inputs,
            torch.stack(features),
            {key: torch.stack(val) if key in self.numerical_cols else torch.tensor(val)
             for key, val in target_batch.items()}
        )

    class NutritionPredictor(nn.Module):
        """BiLSTM for predicting numerical and categorical targets."""
        def __init__(self, embedding_matrix, numerical_cols, categorical_cols, label_vocabs, 
                     feature_dim=4, embedding_dim=300, hid_dim=50, n_layers=2, p=0.3):
            super().__init__()
            self.numerical_cols = numerical_cols
            self.categorical_cols = categorical_cols
            embedding_tensor = torch.tensor(embedding_matrix, dtype=torch.float)
            self.embedding = nn.Embedding.from_pretrained(embedding_tensor, freeze=False)
            self.lstm = nn.LSTM(
                input_size=embedding_dim,
                hidden_size=hid_dim,
                num_layers=n_layers,
                bidirectional=True,
                batch_first=True
            )
            self.drop = nn.Dropout(p)
            self.drop_emb = nn.Dropout(p/1.5)
            self.bn1 = nn.BatchNorm1d(num_features=hid_dim * 2)
            self.feature_fc = nn.Linear(feature_dim, hid_dim)
            self.output_heads = nn.ModuleDict({
                col: nn.Linear(hid_dim * 3, 1 if col in numerical_cols else len(label_vocabs[col]))
                for col in numerical_cols + categorical_cols
            })

        def forward(self, inputs, features):
            embeds = self.embedding(inputs)
            embeds_drop = self.drop_emb(embeds)
            lstm_out, _ = self.lstm(embeds_drop)
            lstm_out = self.drop(lstm_out[:, -1, :])
            lstm_out = self.bn1(lstm_out)
            feature_out = torch.relu(self.feature_fc(features))
            combined = torch.cat((lstm_out, feature_out), dim=1)
            outputs = {}
            for col in self.numerical_cols:
                outputs[col] = self.output_heads[col](combined).squeeze(-1)
            for col in self.categorical_cols:
                outputs[col] = torch.softmax(self.output_heads[col](combined), dim=1)
            return outputs

    def train_model(self, model, train_loader, valid_loader, num_epochs=100, eval_every=None):
        """Train with dynamic loss, gradient clipping, and NaN checks."""
        if eval_every is None:
            eval_every = len(train_loader) // 2
        mse_loss = nn.MSELoss()
        ce_loss = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.0001)
        running_loss = valid_running_loss = 0.0
        global_step = 0
        train_loss_list = []
        valid_loss_list = []

        model.train()
        for epoch in range(num_epochs):
            for inputs, features, targets in train_loader:
                inputs, features = inputs.to(self.device), features.to(self.device)
                targets = {k: v.to(self.device) for k, v in targets.items()}
                outputs = model(inputs, features)
                loss = 0.0
                for col in self.numerical_cols:
                    if torch.isnan(outputs[col]).any() or torch.isnan(targets[col]).any():
                        print(f"NaN detected in {col} outputs or targets at step {global_step}")
                        continue
                    loss += mse_loss(outputs[col], targets[col])
                for col in self.categorical_cols:
                    loss += ce_loss(outputs[col], targets[col])

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                running_loss += loss.item() if not torch.isnan(loss) else 0
                global_step += 1

                if global_step % eval_every == 0:
                    model.eval()
                    with torch.no_grad():
                        valid_loss = 0.0
                        valid_batches = 0
                        for inputs, features, targets in valid_loader:
                            inputs, features = inputs.to(self.device), features.to(self.device)
                            targets = {k: v.to(self.device) for k, v in targets.items()}
                            outputs = model(inputs, features)
                            loss = 0.0
                            for col in self.numerical_cols:
                                if torch.isnan(outputs[col]).any() or torch.isnan(targets[col]).any():
                                    print(f"NaN detected in {col} validation at step {global_step}")
                                    continue
                                loss += mse_loss(outputs[col], targets[col])
                            for col in self.categorical_cols:
                                loss += ce_loss(outputs[col], targets[col])
                            valid_loss += loss.item() if not torch.isnan(loss) else 0
                            valid_batches += 1
                    valid_loss = valid_loss / valid_batches if valid_batches > 0 else float('nan')

                    train_loss_list.append(running_loss / eval_every)
                    valid_loss_list.append(valid_loss)
                    running_loss = 0.0
                    model.train()

                    print(f'Epoch [{epoch+1}/{num_epochs}], Step [{global_step}] - '
                          f'Train Loss: {train_loss_list[-1]:.4f}, Valid Loss: {valid_loss:.4f}')

    def evaluate(self, model, test_loader):
        """Evaluate each column: RMSE for numerical, F1-score for categorical."""
        model.eval()
        results = {}
        true_vals = {col: [] for col in self.numerical_cols + self.categorical_cols}
        pred_vals = {col: [] for col in self.numerical_cols + self.categorical_cols}

        with torch.no_grad():
            for inputs, features, targets in test_loader:
                inputs, features = inputs.to(self.device), features.to(self.device)
                targets = {k: v.to(self.device) for k, v in targets.items()}
                outputs = model(inputs, features)
                for col in self.numerical_cols:
                    pred = outputs[col].cpu().numpy()
                    true = targets[col].cpu().numpy()
                    if np.isnan(pred).any() or np.isnan(true).any():
                        print(f"NaN detected in {col} predictions or targets during evaluation")
                        continue
                    # Denormalize for RMSE calculation
                    pred = self.target_scaler.inverse_transform(pred.reshape(-1, 1)).flatten()
                    true = self.target_scaler.inverse_transform(true.reshape(-1, 1)).flatten()
                    true_vals[col].extend(true)
                    pred_vals[col].extend(pred)
                for col in self.categorical_cols:
                    true_vals[col].extend(targets[col].cpu().numpy())
                    pred_vals[col].extend(torch.argmax(outputs[col], dim=1).cpu().numpy())

        # Per-column metrics
        for col in self.numerical_cols:
            if not true_vals[col]:
                results[col] = "Skipped due to NaN values"
                continue
            rmse = np.sqrt(metrics.mean_squared_error(true_vals[col], pred_vals[col]))
            results[col] = f"RMSE: {rmse:.4f}"
        for col in self.categorical_cols:
            if not true_vals[col]:
                results[col] = "Skipped due to insufficient data"
                continue
            f1 = metrics.f1_score(true_vals[col], pred_vals[col], average='weighted')
            results[col] = f"F1-Score: {f1:.4f}"

        print("Evaluation Metrics:")
        for col, metric in results.items():
            print(f"{col}: {metric}")
        return results

    def run(self):
        """Execute the prediction pipeline."""
        data = self.load_data()
        self.analyze_columns(data)
        vocab = self.build_vocab(data['tokens'])
        data['input_ids'] = data['tokens'].apply(lambda x: self.tokens_to_ids(x, vocab))

        for col in self.categorical_cols:
            self.label_vocabs[col] = self.build_label_vocab(data[col])
            data[f'{col}_id'] = data[col].apply(lambda x: self.labels_to_ids(x, self.label_vocabs[col]))

        data = compute_tag_counts(data)
        print("Data shape after tagging:", data.shape)

        features = data[self.feature_cols + ['brands_id', 'countries_tags_id']].values
        train_size = int(0.8 * len(data))
        train_df = data[:train_size]
        valid_df = data[train_size:]
        train_features = features[:train_size]
        valid_features = features[train_size:]

        train_targets = {col: train_df[col].values if col in self.numerical_cols else train_df[f'{col}_id'].values
                         for col in self.numerical_cols + self.categorical_cols}
        valid_targets = {col: valid_df[col].values if col in self.numerical_cols else valid_df[f'{col}_id'].values
                         for col in self.numerical_cols + self.categorical_cols}

        train_dataset = self.NutritionDataset(train_df['input_ids'].tolist(), train_features, train_targets, self.numerical_cols)
        valid_dataset = self.NutritionDataset(valid_df['input_ids'].tolist(), valid_features, valid_targets, self.numerical_cols)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=self.collate_fn)
        valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False, collate_fn=self.collate_fn)
        print("DataLoader creation complete.")

        embedding_matrix = self.load_glove_embeddings(vocab)
        model = self.NutritionPredictor(
            embedding_matrix, self.numerical_cols, self.categorical_cols, self.label_vocabs,
            feature_dim=len(self.feature_cols) + 2
        ).to(self.device)

        self.train_model(model, train_loader, valid_loader, num_epochs=100)
        self.evaluate(model, valid_loader)

if __name__ == "__main__":
    classifier = NutritionalEntityClassifier(target_columns=[
        'energy_100g', 'fat_100g', 'proteins_100g', 'carbohydrates_100g',
        'nutriscore_grade', 'pnns_groups_1'
    ])
    classifier.run()