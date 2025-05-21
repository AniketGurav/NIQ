import pandas as pd
import numpy as np
import spacy
import torch
import torch.nn as nn
import torch.optim as optim

"""
try:
    import torchtext

    from torchtext.legacy.data import Field, TabularDataset, BucketIterator
    from torchtext.legacy.data import LabelField
except ImportError:
    print("torchtext.legacy not found, using torchtext instead.")
""" 

""" 
    preprocess.py: this creates a csv file with the text and labels
""" 

import time
from sklearn.preprocessing import LabelEncoder
torch.cuda.empty_cache()
import nltk
from nltk.tokenize import word_tokenize
import pandas as pd
from collections import Counter


import os
import pandas as pd
import json
# file_path = os.path.join(source_path, 'en_train_set.csv')

source_path = './genData' # en_train_set.csv
file_path = os.path.join(source_path, 'en_train_set.csv')
data = pd.read_csv(file_path)
data = data.drop(columns=['product_id'])
print("Data shape:", data.shape)
print("Data columns:", data.columns)

def decode_labels(encoded_labels, label_vocab): return [label_vocab[str(code)] for code in encoded_labels]


# Create label mapping JSONs only if not present
g1_json_path = os.path.join(source_path, 'labels_G1_code_reference.json')
g2_json_path = os.path.join(source_path, 'labels_G2_code_reference.json')

if not os.path.exists(g1_json_path) or not os.path.exists(g2_json_path):
    print("Generating label mapping files...")

    # Extract unique labels
    unique_g1 = data['pnns_groups_1'].dropna().unique()
    unique_g2 = data['pnns_groups_2'].dropna().unique()

    # Sort for consistency
    unique_g1 = sorted(unique_g1)
    unique_g2 = sorted(unique_g2)

    # Create label → index mappings
    label_map_g1 = {label: idx for idx, label in enumerate(unique_g1)}
    label_map_g2 = {label: idx for idx, label in enumerate(unique_g2)}

    # Save JSONs
    with open(g1_json_path, 'w') as f:
        json.dump(label_map_g1, f, indent=2)

    with open(g2_json_path, 'w') as f:
        json.dump(label_map_g2, f, indent=2)

    print("Label mapping files created.")
else:
    print("Label mapping files already exist.")

with open(r'./genData/labels_G1_code_reference.json') as json_file:
    le_G1 = json.load(json_file)
with open(r'./genData/labels_G2_code_reference.json') as json_file:
    le_G2 = json.load(json_file)

spacy_en = spacy.load('en_core_web_trf')
def tokenizer(text): return [tok.text for tok in spacy_en.tokenizer(text)]

#print("Label mapping files loaded.",le_G1)

device = torch.device('cuda:0' if torch.cuda.is_available else 'cpu')


# Set the custom NLTK data path first
nltk.data.path.append("/cluster/datastore/aniketag/allData/nltk_data")

# Then download punkt to that path

if os.path.exists("/cluster/datastore/aniketag/allData/nltk_data/tokenizers/punkt"):
    nltk.download('punkt', download_dir="/cluster/datastore/aniketag/allData/nltk_data")


# Set NLTK data path for custom directory (no sudo required)
nltk.data.path.append("/cluster/datastore/aniketag/allData/nltk_data")


from collections import Counter
import re

def safe_tokenize(text):
    try:
        #return text.lower().split()  # simple whitespace tokenizer
        return re.findall(r"\b\w+\b", str(text).lower())

    except Exception:
        return []

data['tokens'] = data['text'].apply(safe_tokenize)

#print("Tokenization complete.",data['tokens']) 
#data['tokens'] = data['text'].str.lower().apply(lambda x: word_tokenize(str(x)))

# Tokenize
#data['tokens'] = data['text'].str.lower().apply(word_tokenize)

# Build vocab
counter = Counter(token for tokens in data['tokens'] for token in tokens)
vocab = {word: idx+2 for idx, (word, _) in enumerate(counter.most_common())}
vocab['<pad>'] = 0
vocab['<unk>'] = 1

# Convert tokens to index sequences
def tokens_to_ids(tokens):
    return [vocab.get(token, vocab['<unk>']) for token in tokens]

data['input_ids'] = data['tokens'].apply(tokens_to_ids)

#print("data['input_ids']:", data['input_ids'])

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data['label_G1'] = le.fit_transform(data['pnns_groups_1'])
data['label_G2'] = le.fit_transform(data['pnns_groups_2'])


train_df = data #pd.read_csv(file_path)#f'{source_path}/en_train_set.csv')
valid_df = data #pd.read_csv(file_path)#f'{source_path}/en_test_set.csv')


#train_df = train_df.drop(columns=['product_id'])
#valid_df = valid_df.drop(columns=['product_id'])

train_df.rename(columns={'text': 'ingredients_text'}, inplace=True)

print("Train.columns:", train_df.columns)
print("1.Train shape:", train_df.shape)
print("1.Valid shape:", valid_df.shape)

if 0:
    from tagging import compute_tag_counts

    # This function adds new columns like: num_NUT, num_FLA, etc.
    train_df = compute_tag_counts(train_df)
    valid_df = compute_tag_counts(valid_df)


from tagging1 import IngredientTagger  # Adjust if file is renamed differently

# Instantiate the tagger
tagger = IngredientTagger()

# Apply tagging to both training and validation sets
train_df = tagger.tag_dataframe(train_df)
valid_df = tagger.tag_dataframe(valid_df)



print("2.Train shape after tagging:", train_df.shape)
print("2.Valid shape after tagging:", valid_df.shape)

# Tokenization + vocab creation (for train set)
#train_df['tokens'] = train_df['text'].apply(lambda x: str(x).lower().split())

#print("\n\t columns:",train_df.columns )
train_df['tokens'] = train_df['ingredients_text'].apply(safe_tokenize)


from collections import Counter
counter = Counter(token for tokens in train_df['tokens'] for token in tokens)
vocab = {word: idx + 2 for idx, (word, _) in enumerate(counter.most_common())}
vocab['<pad>'] = 0
vocab['<unk>'] = 1

# Convert text to token IDs
def tokens_to_ids(tokens):
    return [vocab.get(tok, vocab['<unk>']) for tok in tokens]

train_df['input_ids'] = train_df['tokens'].apply(tokens_to_ids)
#valid_df['tokens'] = valid_df['text'].apply(lambda x: str(x).lower().split())

valid_df['tokens'] = valid_df['ingredients_text'].apply(safe_tokenize)

valid_df['input_ids'] = valid_df['tokens'].apply(tokens_to_ids)

print("Tokenization and vocab creation complete.")

#print("Train shape:", train_df.head())

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

class TextDataset(Dataset):
    def __init__(self, dataframe):
        self.inputs = dataframe['input_ids'].tolist()
        self.labels = dataframe['label_G1'].tolist()
        self.labels2 = dataframe['label_G2'].tolist()
        self.extra_feats = dataframe[[col for col in dataframe.columns if col.startswith('num_')]].values


    def __getitem__(self, idx):
        #return torch.tensor(self.inputs[idx]), torch.tensor(self.labels[idx])
        #return torch.tensor(self.inputs[idx], dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long), torch.tensor(self.labels2[idx], dtype=torch.long)

        return (
                torch.tensor(self.inputs[idx], dtype=torch.long),
                torch.tensor(self.labels[idx], dtype=torch.long),
                torch.tensor(self.labels2[idx], dtype=torch.long),
                torch.tensor(self.extra_feats[idx], dtype=torch.float)
            )


    def __len__(self):
        return len(self.inputs)
"""
def collate_fn(batch):
    inputs, labels, labels2 = zip(*batch)
    padded_inputs = pad_sequence(inputs, batch_first=True, padding_value=0)
    return padded_inputs, torch.tensor(labels), torch.tensor(labels2)
"""

def collate_fn(batch):
    inputs, labels, labels2, extras = zip(*batch)
    padded_inputs = pad_sequence(inputs, batch_first=True, padding_value=0)
    return padded_inputs, torch.tensor(labels), torch.tensor(labels2), torch.stack(extras)

train_dataset = TextDataset(train_df)
valid_dataset = TextDataset(valid_df)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

print("DataLoader creation complete.")
print("Train shape:",len(train_loader.dataset))
print("Valid shape:",len(valid_loader.dataset))


"""
# Tokenize already done: data['tokens']
counter = Counter(tok for toks in train_df['tokens'] for tok in toks)

"""

# Count all tokens from your tokenized train set
counter = Counter(tok for toks in train_df['tokens'] for tok in toks)

# ✅ Build vocab only ONCE with min_freq=5
min_freq = 5
#vocab = {word: idx + 2 for idx, (word, freq) in enumerate(counter.items()) if freq >= min_freq}

vocab = {word: idx+2 for idx, (word, _) in enumerate(counter.most_common())}


vocab['<pad>'] = 0
vocab['<unk>'] = 1

def tokens_to_ids(tokens):
    return [vocab.get(tok, vocab['<unk>']) for tok in tokens]

train_df['input_ids'] = train_df['tokens'].apply(tokens_to_ids)
valid_df['input_ids'] = valid_df['tokens'].apply(tokens_to_ids)


import numpy as np

embedding_dim = 300
#glove_path = glove_path+'/glove.6B.300d.txt'  # download from: https://nlp.stanford.edu/projects/glove/

# Load GloVe into dictionary
glove = {}
glove_path = '/cluster/datastore/aniketag/allData/NIQ/glove.6B.300d.txt'

print("Loading GloVe vectors...",glove_path)

with open(glove_path, 'r', encoding='utf-8') as f:
    for line in f:
        values = line.strip().split()
        word = values[0]
        vec = np.array(values[1:], dtype='float32')
        glove[word] = vec

    print("GloVe loaded successfully.")



# Create embedding matrix for words in our vocab
embedding_matrix = np.zeros((len(vocab), embedding_dim))

print("Creating embedding matrix...",len(vocab), embedding_dim) 

#print("Vocab :",vocab)


for word, idx in vocab.items():
    #print("Word:",word,"Idx:",idx)
    if word in glove:
        embedding_matrix[idx] = glove[word]
    else:
        embedding_matrix[idx] = np.random.normal(scale=0.6, size=(embedding_dim,))


import torch

class LSTMG1(nn.Module):
    def __init__(self,embedding_matrix, embedding_dim=300, hid_dim=50, n_layers=2, p=0.3, n_classes=9, extra_feat_dim=10):
        super(LSTMG1, self).__init__()

        embedding_tensor = torch.tensor(embedding_matrix, dtype=torch.float)
        self.embedding = nn.Embedding.from_pretrained(embedding_tensor, freeze=False)

        #self.embedding = nn.Embedding.from_pretrained(TEXT.vocab.vectors, freeze=False)
        self.lstm = nn.LSTM(
            input_size = embedding_dim, 
            hidden_size = hid_dim, 
            num_layers = n_layers,
            bidirectional = True,
            batch_first = True)
        self.drop = nn.Dropout(p)
        self.drop_emb = nn.Dropout(p/1.5)
        self.bn1 = nn.BatchNorm1d(num_features=hid_dim)
        self.hid_out = nn.Linear(hid_dim, n_classes)
    
        self.extra_fc = nn.Linear(extra_feat_dim, hid_dim)  # add this

        #print("hid_dim.shape:", hid_dim)
        self.combined_out = nn.Linear(60, n_classes)  # replace self.hid_out



    def forward(self, inputs,extra_feats):
        embeds = self.embedding(inputs)
        embeds_drop = self.drop(embeds)
        outputs, (h_n, c_n) = self.lstm(embeds_drop)
        x = self.drop(h_n[0])
        x = self.bn1(x)
        x = self.hid_out(x)

        x_extra = torch.relu(self.extra_fc(extra_feats))  # [batch, hid_dim]

        x_combined = torch.cat((x, x_extra), dim=1)  # [batch, hid_dim*2]

        #print("x_combined.shape:",x_combined.shape,"\t x.shape:",x.shape,"\t x_extra.shape:",x_extra.shape)

        return self.combined_out(x_combined)

def train(model,
        model2,
        optimizer,
        criterion = nn.CrossEntropyLoss(),
        train_loader = train_loader,
        valid_loader = valid_loader,
        num_epochs = 100,
        eval_every = len(train_loader) // 2,
        best_valid_loss = float("Inf")):
          
        running_loss = 0.0
        running_loss2 = 0.0

        valid_running_loss = 0.0
        valid_running_loss2 = 0.0

        global_step = 0
        train_loss_list = []
        valid_loss_list = []
        global_steps_list = []

        train_loss_list2 = []
        valid_loss_list2 = []
        global_steps_list2 = []


        model.train()
        model2.train()
        optimizer2 = optimizer

        for epoch in range(num_epochs):
            for batch in train_loader:

                #data = batch.text.to(device)           
                #labels = batch.label.to(device)

                #data, labels,labels2 = batch

                data, labels, labels2, extra_feats = batch

                data = data.to(device)
                labels = labels.to(device)
                labels2 = labels2.to(device)
                extra_feats = extra_feats.to(device)


                output = model(data,extra_feats)
                output2 = model2(data, extra_feats)
                
    
                loss = criterion(output, labels)
                loss2 = criterion(output2, labels)


                #print("output.shape::",output.shape," labels.shape:",labels.shape)
                #print("Loss:",loss.item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                optimizer2.zero_grad()
                loss2.backward()
                optimizer2.step()


                #update running vals
                running_loss += loss.item()
                running_loss2 += loss2.item()


                global_step += 1
                
                #Eval step
                
                if global_step % eval_every == 0: model.eval()
                
                # validation loop
                with torch.no_grad():

                    for batch in valid_loader:
                        #data = batch.text.to(device)
                        #labels = batch.label.to(device)

                        #data, labels,labels2 = batch

                        data, labels, labels2, extra_feats = batch

                        data = data.to(device)
                        labels = labels.to(device)
                        labels2 = labels2.to(device)
                        extra_feats = extra_feats.to(device)


                        output = model(data, extra_feats)
                        output2 = model2(data, extra_feats)  # assuming both models use the same inputs
                        
            
                        loss = criterion(output, labels)
                        loss2 = criterion(output2, labels)
                        
                        valid_running_loss += loss.item()
                        valid_running_loss2 += loss2.item()


                average_train_loss = running_loss / eval_every
                average_train_loss2 = running_loss2 / eval_every

                average_valid_loss = valid_running_loss / len(valid_loader)
                average_valid_loss2 = valid_running_loss2 / len(valid_loader)

                train_loss_list.append(average_train_loss)
                valid_loss_list.append(average_valid_loss)
                global_steps_list.append(global_step)

                train_loss_list2.append(average_train_loss2)
                valid_loss_list2.append(average_valid_loss2)
                global_steps_list2.append(global_step)

                # resetting running values
                running_loss = 0.0
                valid_running_loss = 0.0
                model.train()

                # print progress
                
                print('Epoch [{}/{}], Step [{}/{}] - Train Loss: {:.4f}, Valid Loss: {:.4f}'
                .format(epoch+1, num_epochs, global_step, num_epochs*len(train_loader),
                average_train_loss, average_valid_loss))
                print('-'*50)
                
                if best_valid_loss > average_valid_loss:
                    best_valid_loss = average_valid_loss
            print('_'*50)

def get_classification_report(y_test, y_pred, sortby='precision', model='model'):
    """Return a classification report as pd.DataFrame, sorted and labeled by model name."""
    from sklearn import metrics
    import pandas as pd

    # Generate classification report
    report = metrics.classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    # Convert to DataFrame
    df = pd.DataFrame(report).transpose()

    # Keep only numeric rows (drop 'accuracy' row)
    df = df[df.index.str.isnumeric() | df.index.isin(['macro avg', 'weighted avg'])]

    # Sort safely (if sortby exists)
    if sortby in df.columns:
        df = df.sort_values(by=sortby, ascending=False)

    # Rename columns with model prefix
    df.rename(columns={col: f"{model}_{col}" for col in df.columns}, inplace=True)

    return df.round(2)


def evaluate(model,model2, test_iter, le, device='cpu'):
    """Evaluate a model on the test set and return classification report."""
    import numpy as np
    import torch

    y_true = []
    y_preds = []

    y_true2 = []
    y_preds2 = []

    model.to(device)
    model.eval()  # ensure model is in eval mode

    model2.to(device)
    model2.eval()  # ensure model is in eval mode


    with torch.no_grad():
        for batch in test_iter:
            #data = batch.text.to(device)
            #labels = batch.label.to(device)

            data, labels, labels2, extra_feats = batch
            data = data.to(device)
            labels = labels.to(device)
            labels2 = labels.to(device)
            extra_feats = extra_feats.to(device)

            outputs = model(data, extra_feats)
            _, predicted = torch.max(outputs.data, 1)

            y_preds.extend(predicted.cpu().numpy())
            y_true.extend(labels.cpu().numpy())


            outputs2 = model2(data, extra_feats)
            _, predicted2 = torch.max(outputs2.data, 1)

            y_preds2.extend(predicted2.cpu().numpy())
            y_true2.extend(labels2.cpu().numpy())



        from sklearn.metrics import accuracy_score
        accuracy = accuracy_score(y_true, y_preds)
        print(f"Accuracy1: {accuracy:.4f}")

        accuracy2 = accuracy_score(y_true2, y_preds2)
        print(f"Accuracy2: {accuracy2:.4f}")



    # Convert predictions back to original labels if label encoder provided
    if le is not None:
        y_true = le.inverse_transform(y_true)
        y_preds = le.inverse_transform(y_preds)

    report = get_classification_report(y_true, y_preds, model='BI_LSTM_G1')
    return report


#net_G1 = LSTMG1(embedding_matrix,embedding_dim=300, n_layers=3, n_classes=100).to(device)

extra_feat_dim = len([col for col in train_df.columns if col.startswith('num_')])

print("Extra feature dimension:", extra_feat_dim)
net_G1 = LSTMG1(embedding_matrix, embedding_dim=300, n_layers=3, n_classes=10, extra_feat_dim=extra_feat_dim).to(device)


net_G2 = LSTMG1(embedding_matrix, embedding_dim=300, n_layers=3, n_classes=10, extra_feat_dim=extra_feat_dim).to(device)
optimizer = optim.Adam(net_G1.parameters(), lr=0.001)

print("Label max:", train_df['label_G1'].max())
print("Unique labels:", train_df['label_G1'].unique())

num_classes = 10  # should match model output
assert train_df['label_G1'].max() < num_classes, "⚠️ Label out of bounds for CrossEntropyLoss!"


train(model=net_G1,model2=net_G2, optimizer=optimizer, num_epochs=100)

report = evaluate(net_G1,net_G2, valid_loader, le) #le_G1

print("Classification report:\n", report)
