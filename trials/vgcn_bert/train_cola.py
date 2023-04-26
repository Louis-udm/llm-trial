import os
from pathlib import Path
import pandas as pd
from sklearn.utils import shuffle
import numpy as np
import time
import random
import dill
import torch
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import classification_report
from sklearn.metrics import f1_score

import transformers as tfr
from transformers import AdamW
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.vgcn_bert.modeling_graph import WordGraph,_normalize_adj

random.seed(44)
np.random.seed(44)
torch.manual_seed(44)

cuda_yes = torch.cuda.is_available()
if cuda_yes:
    torch.cuda.manual_seed_all(44)
device = torch.device("cuda:0" if cuda_yes else "cpu")


# set environment variable to use local models
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# specify path to local model files
model_path = (
    "/tmp/local-huggingface-models/zhibinlu_vgcn-distilbert-base-uncased"
)

# load tokenizer and model
tokenizer = tfr.AutoTokenizer.from_pretrained(model_path)

""" 
Load cola dataset
"""

ds_path = "./data/CoLA"

valid_data_taux = 0.05
test_data_taux = 0.10

label2idx = {"0": 0, "1": 1}
idx2label = {0: "0", 1: "1"}
train_valid_df = pd.read_csv(
    os.path.join(ds_path, "train.tsv"), encoding="utf-8", header=None, sep="\t"
)
train_valid_df=train_valid_df[[0,1,3]] # 0-ref, 1-label, 3-sentence
train_valid_df = shuffle(train_valid_df)
# use dev set as test set, because we can not get the ground true label of the real test set.
test_df = pd.read_csv(
    os.path.join(ds_path, "dev.tsv"), encoding="utf-8", header=None, sep="\t"
)
test_df=test_df[[0,1,3]]

train_valid_size = train_valid_df[1].count()
valid_size = int(train_valid_size * valid_data_taux)
train_size = train_valid_size - valid_size
test_size = test_df[1].count()
print("CoLA train_valid Total:", train_valid_size, "test Total:", test_size)
# df = pd.concat((train_valid_df, test_df))
# corpus = df[3]
# y = df[1].values  # y.as_matrix()
# confidence
# y_prob = np.eye(len(y), len(label2idx))[y]
# corpus_size = len(y)

y_train_valid = train_valid_df[1].values
y_test = test_df[1].values
y_prob_train_valid = np.eye(len(y_train_valid), len(label2idx))[y_train_valid]
y_prob_test = np.eye(len(y_test), len(label2idx))[y_test]


"""
build/Load wgraph
"""

cola_wgraph_path = "/tmp/vgcn-bert/cola_wgraph_win1000_nosw_minifreq2.pkl"
if os.path.exists(cola_wgraph_path):
    with open(cola_wgraph_path, "rb") as f:
        wgraph = dill.load(f)
else:
    wgraph=WordGraph(rows=train_valid_df[3], tokenizer=tokenizer,window_size=1000, remove_stopwords=False)
    with open(cola_wgraph_path, "wb") as f:
        dill.dump(wgraph,f)

# Zero ratio before and after normalization is the same.
print(
    "  Zero ratio(?>66%%) of the graph matrix: %.8f"
    % (
        100
        * (
            1
            - wgraph.adjacency_matrix.count_nonzero()
            / (
                wgraph.adjacency_matrix.shape[0]
                * wgraph.adjacency_matrix.shape[1]
            )
        )
    )
)

"""
Prepare dataset
"""
class ColaDataset(Dataset):
    def __init__(self, corpus, y, y_prob, tokenizer,max_len=512):
        self.corpus = corpus
        self.y = y
        self.y_prob = y_prob
        assert len(self.corpus) == len(self.y) == len(self.y_prob)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        sentence = self.corpus.iloc[idx]
        # sentence = self.corpus[idx]
        y = self.y[idx]
        y_prob = self.y_prob[idx]
        # https://huggingface.co/transformers/v3.5.1/internal/tokenization_utils.html#transformers.tokenization_utils_base.PreTrainedTokenizerBase.encode_plus
        inputs = self.tokenizer.encode_plus(
            text=sentence,
            text_pair=None,
            add_special_tokens=True,
            max_length=self.max_len,
            # pad_to_max_length=True,
            padding="max_length",
            # padding="longest",
            # return_token_type_ids=True,
            truncation=True,
            return_tensors="pt",
        )
        # inputs = {k: v.to(device) for k, v in inputs.items()}
        # y = torch.tensor(y, dtype=torch.long).to(device)
        # y_prob = torch.tensor(y_prob, dtype=torch.float).to(device)
        return inputs,y,y_prob
    
MAX_LEN = 512-16
TRAIN_BATCH_SIZE = 2
VALID_BATCH_SIZE = 32
TOTAL_EPOCH = 9
LEARNING_RATE = 1e-05 # old 8e-6
DROPOUT_RATE = 0.2  # 0.5 # Dropout rate (1 - keep probability).
L2_DECAY = 0.001
DO_LOWER_CASE = True

train_dataloader = DataLoader(dataset=ColaDataset(train_valid_df[3][:train_size], y_train_valid[:train_size], y_prob_train_valid[:train_size], tokenizer,MAX_LEN), batch_size=TRAIN_BATCH_SIZE, shuffle=False)
valid_dataloader = DataLoader(dataset=ColaDataset(train_valid_df[3][train_size:], y_train_valid[train_size:], y_prob_train_valid[train_size:], tokenizer,MAX_LEN), batch_size=VALID_BATCH_SIZE, shuffle=False)
test_dataloader = DataLoader(dataset=ColaDataset(test_df[3], y_test, y_prob_test, tokenizer,MAX_LEN), batch_size=VALID_BATCH_SIZE, shuffle=False)

"""
Init Model
"""
torch_graph = wgraph.to_torch_sparse()

# model = tfr.AutoModel.from_pretrained(model_path)
model = tfr.AutoModelForSequenceClassification.from_pretrained(
    model_path,
    [torch_graph],
    [wgraph.wgraph_id_to_tokenizer_id_map],
)
model.to(device)
# model.vgcn_bert.embeddings.vgcn.set_transparent_parameters()


"""
Train
"""

def evaluate(model, dataloader):
    model.eval()
    total_loss = 0
    total_accuracy = 0
    total_preds = []
    total_y = []
    for batch in dataloader:
        inputs, y, y_prob = batch
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs:SequenceClassifierOutput = model(**inputs)
            loss, logits, hidden_states, attentions = outputs
        logits = logits.detach().cpu().numpy()
        # label_ids = y.to("cpu").numpy()
        total_loss += loss.item()
        total_preds.append(logits.tolist())
        total_y.append(y.tolist())
    avg_loss = total_loss / len(dataloader)
    total_preds = np.concatenate(total_preds, axis=0)
    total_y = np.concatenate(total_y, axis=0)
    return avg_loss, total_preds, total_y


def train(model, train_dataloader, valid_dataloader):
    for epoch in range(TOTAL_EPOCH):
        model.train()
        total_loss = 0
        for step, batch in enumerate(train_dataloader):
            inputs, y, y_prob = batch
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs:SequenceClassifierOutput = model(**inputs)
            loss, logits, hidden_states, attentions = outputs
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()
        avg_train_loss = total_loss / len(train_dataloader)
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        t0=time.time()
        avg_valid_loss, _, _ = evaluate(model, valid_dataloader)
        print("  Average valid loss: {0:.2f}".format(avg_valid_loss))
        print("  Validation took: {:}".format(format_time(time.time() - t0)))
    return avg_train_loss, avg_valid_loss


optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

train_start = time.time()
train(model, train_dataloader, valid_dataloader)

"""
Test
"""
avg_test_loss, test_preds, test_y = evaluate(model, test_dataloader)

"""
Examples
"""
# example sentence to classify
sentences = ["I really enjoyed this movie!", "A boring movie."]
# sentence = "fdsheibif gjetrsg, suisfdsbewg monfjei, Je suis là! je suis à Montréal! C'est ce que je veux. J'ai vraiment apprécié ce film! C'est ça!"

# tokenize sentence
# DistilBertTokenizerFast
inputs = tokenizer(
    sentences,
    max_length=512,
    # padding="max_length",
    # padding="do_not_pad",
    padding=True,
    truncation=True,
    return_tensors="pt",
)
print(inputs)
ids = inputs["input_ids"].view(-1)
print(ids.view(-1))
# 101 cls, 102 sep, 0 pad
print(tokenizer.convert_ids_to_tokens(ids))
print(tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(ids)))

# # classify sentence
inputs = {k: v.to(device) for k, v in inputs.items()}
outputs = model(**inputs)
predictions = outputs.logits.argmax(dim=1)

# # print predicted label
labels = ["negative", "positive"]
print("Prediction:", [labels[p] for p in predictions])

"""
save weights
"""
# model.save_pretrained("/tmp/local-huggingface-models/zhibinlu_vgcn-distilbert-base-uncased-all")
