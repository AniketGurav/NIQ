import pandas as pd
import numpy as np
import spacy
import torch
import torch.nn as nn
import torch.optim as optim
import time
from sklearn.preprocessing import LabelEncoder
import nltk
from nltk.tokenize import word_tokenize
from collections import Counter
import os
import json
import re
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn import metrics
from tagging import compute_tag_counts

class NutritionalEntityClassifier:
    """
    Trains a BiLSTM to tag nutritional entities (e.g., nutrient, quantity, unit) in ingredients_text.

    Why: Extracts structured entities like "Protein: 10 g" via sequence labeling, using token and tag features.

    Pseudo Logic:
        - Load tagged data, convert tags to BIO format
        - Tokenize text, build vocab, convert to IDs
        - Create dataset/loader with token IDs, BIO labels, tag counts
        - Load GloVe embeddings for token representations
        - Train BiLSTM for per-token entity tagging
        - Evaluate with F1-score and classification report
    """
    
    def __init__(self, data_path='./genData/tagged_ingredient_output_with_pairs.csv', 
                 glove_path='/cluster/datastore/aniketag/allData/NIQ/glove.6B.300d.txt'):
        """Set up paths, device, and clear CUDA cache."""
        self.data_path = data_path
        self.glove_path = glove_path
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        torch.cuda.empty_cache()
        self.spacy_en = spacy.load('en_core_web_trf')
        nltk.data.path.append("/cluster/datastore/aniketag/allData/nltk_data")
        if os.path.exists("/cluster/datastore/aniketag/allData/nltk_data/tokenizers/punkt"):
            nltk.download('punkt', download_dir="/cluster/datastore/aniketag/allData/nltk_data")

    def load_data(self):
        """Load tagged data, eval token/tags lists, and print shape."""
        data = pd.read_csv(self.data_path)
        data['tokens'] = data['tokens'].apply(eval)  # Convert string lists to actual lists
        data['tags'] = data['tags'].apply(eval)
        print("Data shape:", data.shape)
        return data

    def create_bio_labels(self, data):
        """Convert tags to BIO format for sequence labeling."""
        bio_labels = []
        for tags in data['tags']:
            bio_seq = []
            for i, tag in enumerate(tags):
                if tag in ['NUT', 'QTY', 'UNI']:
                    prefix = 'B' if i == 0 or tags[i-1] != tag else 'I'
                    bio_seq.append(f"{prefix}-{tag}")
                else:
                    bio_seq.append('O')
            bio_labels.append(bio_seq)
        data['bio_labels'] = bio_labels
        return data

    def tokenize(self, text):
        """Tokenize text into clean, lowercase words with regex."""
        try:
            return re.findall(r"\b\w+\b", str(text).lower())
        except Exception:
            return []

    def build_vocab(self, tokens):
        """Build vocab from tokens, add pad/unk tokens."""
        counter = Counter(token for token_list in tokens for token in token_list)
        vocab = {word: idx + 2 for idx, (word, _) in enumerate(counter.most_common())}
        vocab['<pad>'] = 0
        vocab['<unk>'] = 1
        return vocab

    def build_label_vocab(self, bio_labels):
        """Build vocab for BIO labels."""
        unique_labels = sorted(set(label for seq in bio_labels for label in seq))
        return {label: idx for idx, label in enumerate(unique_labels)}

    def tokens_to_ids(self, tokens, vocab):
        """Convert tokens to numeric IDs."""
        return [vocab.get(token, vocab['<unk>']) for token in tokens]

    def labels_to_ids(self, labels, label_vocab):
        """Convert BIO labels to numeric IDs."""
        return [label_vocab[label] for label in labels]

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

    class TextDataset(Dataset):
        """Dataset for token IDs, BIO label IDs, and tag features."""
        def __init__(self, input_ids, label_ids, extra_feats):
            self.inputs = input_ids
            self.labels = label_ids
            self.extra_feats = extra_feats

        def __getitem__(self, idx):
            return (
                torch.tensor(self.inputs[idx], dtype=torch.long),
                torch.tensor(self.labels[idx], dtype=torch.long),
                torch.tensor(self.extra_feats[idx], dtype=torch.float)
            )

        def __len__(self):
            return len(self.inputs)

    def collate_fn(self, batch):
        """Pad sequences and format batch for DataLoader."""
        inputs, labels, extras = zip(*batch)
        padded_inputs = pad_sequence(inputs, batch_first=True, padding_value=0)
        padded_labels = pad_sequence(labels, batch_first=True, padding_value=-100)  # Ignore padding in loss
        return padded_inputs, padded_labels, torch.stack(extras)

    class EntityLSTM(nn.Module):
        """BiLSTM for per-token entity tagging with tag feature integration."""
        def __init__(self, embedding_matrix, embedding_dim=300, hid_dim=50, n_layers=2, p=0.3, n_classes=10, extra_feat_dim=14):
            super().__init__()
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
            self.bn1 = nn.BatchNorm1d(num_features=hid_dim * 2)  # Bidirectional doubles hid_dim
            self.extra_fc = nn.Linear(extra_feat_dim, hid_dim)
            self.combined_out = nn.Linear(hid_dim * 2 + hid_dim, n_classes)

        def forward(self, inputs, extra_feats):
            embeds = self.embedding(inputs)
            embeds_drop = self.drop_emb(embeds)
            lstm_out, _ = self.lstm(embeds_drop)  # [batch, seq_len, hid_dim*2]
            lstm_out = self.drop(lstm_out)
            lstm_out = self.bn1(lstm_out.transpose(1, 2)).transpose(1, 2)
            x_extra = torch.relu(self.extra_fc(extra_feats))  # [batch, hid_dim]
            x_extra = x_extra.unsqueeze(1).expand(-1, lstm_out.size(1), -1)  # [batch, seq_len, hid_dim]
            x_combined = torch.cat((lstm_out, x_extra), dim=2)  # [batch, seq_len, hid_dim*3]
            return self.combined_out(x_combined)  # [batch, seq_len, n_classes]

    def train_model(self, model, train_loader, valid_loader, num_epochs=100, eval_every=None):
        """Train the model, track losses, and print progress."""
        if eval_every is None:
            eval_every = len(train_loader) // 2
        criterion = nn.CrossEntropyLoss(ignore_index=-100)  # Ignore padding labels
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        running_loss = valid_running_loss = 0.0
        global_step = 0
        train_loss_list = []
        valid_loss_list = []
        best_valid_loss = float("Inf")

        model.train()
        for epoch in range(num_epochs):
            for batch in train_loader:
                data, labels, extra_feats = batch
                data, labels, extra_feats = data.to(self.device), labels.to(self.device), extra_feats.to(self.device)

                outputs = model(data, extra_feats)  # [batch, seq_len, n_classes]
                outputs = outputs.view(-1, outputs.size(-1))  # [batch*seq_len, n_classes]
                labels = labels.view(-1)  # [batch*seq_len]
                loss = criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                global_step += 1

                if global_step % eval_every == 0:
                    model.eval()
                    with torch.no_grad():
                        for batch in valid_loader:
                            data, labels, extra_feats = batch
                            data, labels, extra_feats = data.to(self.device), labels.to(self.device), extra_feats.to(self.device)
                            outputs = model(data, extra_feats)
                            outputs = outputs.view(-1, outputs.size(-1))
                            labels = labels.view(-1)
                            loss = criterion(outputs, labels)
                            valid_running_loss += loss.item()

                    average_train_loss = running_loss / eval_every
                    average_valid_loss = valid_running_loss / len(valid_loader)
                    train_loss_list.append(average_train_loss)
                    valid_loss_list.append(average_valid_loss)

                    running_loss = valid_running_loss = 0.0
                    model.train()

                    print(f'Epoch [{epoch+1}/{num_epochs}], Step [{global_step}/{num_epochs*len(train_loader)}] - Train Loss: {average_train_loss:.4f}, Valid Loss: {average_valid_loss:.4f}')
                    print('-' * 50)

                    if best_valid_loss > average_valid_loss:
                        best_valid_loss = average_valid_loss
            print('_' * 50)

    def evaluate(self, model, test_loader, label_vocab):
        """Evaluate model, compute F1-score and classification report."""
        y_true, y_preds = [], []
        model.to(self.device)
        model.eval()
        inv_label_vocab = {v: k for k, v in label_vocab.items()}

        with torch.no_grad():
            for batch in test_loader:
                data, labels, extra_feats = batch
                data, labels, extra_feats = data.to(self.device), labels.to(self.device), extra_feats.to(self.device)
                outputs = model(data, extra_feats)
                _, predicted = torch.max(outputs, dim=2)
                for pred_seq, true_seq in zip(predicted.cpu().numpy(), labels.cpu().numpy()):
                    valid_preds = [inv_label_vocab[p] for p, t in zip(pred_seq, true_seq) if t != -100]
                    valid_true = [inv_label_vocab[t] for t in true_seq if t != -100]
                    y_preds.extend(valid_preds)
                    y_true.extend(valid_true)

        report = metrics.classification_report(y_true, y_preds, output_dict=True, zero_division=0)
        f1_score = report['weighted avg']['f1-score']
        print(f"F1-Score: {f1_score:.4f}")
        df_report = pd.DataFrame(report).transpose()
        df_report = df_report[df_report.index.str.isnumeric() | df_report.index.isin(['macro avg', 'weighted avg'])]
        df_report = df_report.sort_values(by='f1-score', ascending=False)
        df_report.rename(columns={col: f"EntityLSTM_{col}" for col in df_report.columns}, inplace=True)
        return df_report.round(2), f1_score

    def run(self):
        """Execute the entity tagging pipeline."""
        data = self.load_data()
        data = self.create_bio_labels(data)
        data['tokens'] = data['ingredients_text'].apply(self.tokenize)
        vocab = self.build_vocab(data['tokens'])
        label_vocab = self.build_label_vocab(data['bio_labels'])
        data['input_ids'] = data['tokens'].apply(lambda x: self.tokens_to_ids(x, vocab))
        data['label_ids'] = data['bio_labels'].apply(lambda x: self.labels_to_ids(x, label_vocab))
        
        # Split train/valid (80/20 split)
        train_size = int(0.8 * len(data))
        train_df = data[:train_size]
        valid_df = data[train_size:]
        
        # Add tag counts
        train_df = compute_tag_counts(train_df)
        valid_df = compute_tag_counts(valid_df)
        print("Train shape:", train_df.shape)
        print("Valid shape:", valid_df.shape)

        # Prepare dataset with precomputed IDs
        train_dataset = self.TextDataset(
            input_ids=train_df['input_ids'].tolist(),
            label_ids=train_df['label_ids'].tolist(),
            extra_feats=train_df[[col for col in train_df.columns if col.startswith('num_')]].values
        )
        valid_dataset = self.TextDataset(
            input_ids=valid_df['input_ids'].tolist(),
            label_ids=valid_df['label_ids'].tolist(),
            extra_feats=valid_df[[col for col in valid_df.columns if col.startswith('num_')]].values
        )
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=self.collate_fn)
        valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False, collate_fn=self.collate_fn)
        print("DataLoader creation complete.")

        embedding_matrix = self.load_glove_embeddings(vocab)
        n_classes = len(label_vocab)
        extra_feat_dim = len([col for col in train_df.columns if col.startswith('num_')])
        model = self.EntityLSTM(embedding_matrix, embedding_dim=300, n_layers=3, n_classes=n_classes, 
                              extra_feat_dim=extra_feat_dim).to(self.device)

        self.train_model(model, train_loader, valid_loader, num_epochs=100)
        report, f1_score = self.evaluate(model, valid_loader, label_vocab)
        print("Classification report:\n", report)

if __name__ == "__main__":
    classifier = NutritionalEntityClassifier()
    classifier.run()