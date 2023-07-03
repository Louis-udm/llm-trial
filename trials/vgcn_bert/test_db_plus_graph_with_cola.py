import os
from pathlib import Path
import shutil
import pandas as pd
from sklearn.utils import shuffle
import numpy as np
import time
import random
import dill
import torch
import argparse
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import classification_report
from sklearn.metrics import f1_score

import transformers as tfr
from transformers import DistilBertTokenizer,AdamW
# from torch.optim import SparseAdam
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.vgcn_bert.modeling_graph import WordGraph,_normalize_adj


print(f"\nStart Datetime: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")

seed=44
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

cuda_yes = torch.cuda.is_available()
if cuda_yes:
    torch.cuda.manual_seed_all(seed)
# device="cpu"
device = torch.device("cuda:0" if cuda_yes else "cpu")
print(f"Random Seed: {seed}, Device: {device}")

# set environment variable to use local models
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# specify path to local model files
tokenizer_path = "/tmp/local-huggingface-models/hf-maintainers_distilbert-base-uncased"
standard_vb_model_path="/tmp/local-huggingface-models/zhibinlu_vgcn-distilbert-base-uncased"
db_model_path = "/tmp/vgcn-bert/model_savings/cola_db_5e5"
              
print(f"Model Path: {db_model_path} \nModel config:")
with open(os.path.join(db_model_path, "config.json"), "rt") as f:
    print(f.read())

# load tokenizer and model
tokenizer = DistilBertTokenizer.from_pretrained(tokenizer_path)

""" 
Load cola dataset
"""

ds_path = "./data/CoLA"
print(f"\nDataset Path: {ds_path}")

valid_data_taux = 0.05
test_data_taux = 0.10
print(f"Valid Data Taux: {valid_data_taux}, Test Data Taux: {test_data_taux}")

label2idx = {"0": 0, "1": 1}
idx2label = {0: "0", 1: "1"}
train_valid_df = pd.read_csv(
    os.path.join(ds_path, "train.tsv"), encoding="utf-8", header = None, sep="\t"
)
train_valid_df = train_valid_df[[0,1,3]] # 0-ref, 1-label, 3-sentence
train_valid_df = shuffle(train_valid_df)
# use dev set as test set, because we can not get the ground true label of the real test set.
test_df = pd.read_csv(
    os.path.join(ds_path, "dev.tsv"), encoding="utf-8", header = None, sep="\t"
)
test_df = test_df[[0,1,3]]

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
Prepare dataset
"""
class ColaDataset(Dataset):
    def __init__(self, corpus, y, y_prob, tokenizer, max_len=512):
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
        inputs = self.tokenizer(
            text = sentence,
            text_pair = None,
            padding="max_length", 
            max_length = self.max_len, 
            truncation = True,
            return_tensors="pt",
        )
        inputs={k:v[0] for k, v in inputs.items()}
        return inputs, y, y_prob
    
BATCH_SIZE = 32
MAX_LEN = 512-16
print(f"Test Batch Size: {BATCH_SIZE}")

valid_dataloader = DataLoader(
    dataset = ColaDataset(
        train_valid_df[3][train_size:], 
        y_train_valid[train_size:], 
        y_prob_train_valid[train_size:], 
        tokenizer,
        MAX_LEN
    ), batch_size = BATCH_SIZE, shuffle = False)
test_dataloader = DataLoader(
    dataset = ColaDataset(
        test_df[3], 
        y_test, 
        y_prob_test, 
        tokenizer,
        MAX_LEN
    ), batch_size = BATCH_SIZE, shuffle = False)

"""
Load wgraph
"""
cola_wgraph_path = "/tmp/vgcn-bert/wgraphs/cola_wgraph_win1000_nosw_minifreq2.pkl"
print(f"Word Graph Path: {cola_wgraph_path}")

# if os.path.exists(cola_wgraph_path):
with open(cola_wgraph_path, "rb") as f:
    wgraph = dill.load(f)
# else:
#     window_size=1000
#     remove_stopwords = False
#     print(f"Build Word Graph, winidow_size={window_size}, remove_stopwords={remove_stopwords}")
#     wgraph = WordGraph(
#         rows = train_valid_df[3], 
#         tokenizer = tokenizer,
#         window_size = window_size, 
#         remove_stopwords = remove_stopwords
#     )
#     with open(cola_wgraph_path, "wb") as f:
#         dill.dump(wgraph, f)

torch_graph = wgraph.to_torch_sparse()

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
Init Model
"""
from os import path
state_dict=torch.load(path.join(db_model_path,"pytorch_model.bin"),map_location=torch.device(device))
new_model_path = f"{db_model_path}_renamed_to_vb"
if not path.exists(new_model_path):
    os.mkdir(new_model_path)
    # copy standard_vb_model_path/config.json to new_model_path
    shutil.copyfile(path.join(standard_vb_model_path,"config.json"), path.join(new_model_path,"config.json"))
new_tensors={}
for k,v in state_dict.items():
    if k.startswith("distilbert"):
        new_tensors[f"vgcn_bert.{k[11:]}"]=v
    else:
        new_tensors[k]=v
torch.save(new_tensors, path.join(new_model_path,"pytorch_model.bin"))

model = tfr.AutoModelForSequenceClassification.from_pretrained(
    new_model_path,
    [torch_graph],
    [wgraph.wgraph_id_to_tokenizer_id_map],
)

print(f"\nVGCN weights transparant setting:{(model.vgcn_bert.embeddings.vgcn.W_vh_list[0]==1).all()}")
print(f"VGCN fc transparant setting:{(model.vgcn_bert.embeddings.vgcn.fc_hg.weight==1).all()}")

model.to(device)
# model.vgcn_bert.embeddings.vgcn.set_transparent_parameters()

# get GPU memory usage in Mb for the current device
def get_gpu_memory_usage():
    return max([torch.cuda.memory_allocated(),torch.cuda.max_memory_allocated(), torch.cuda.memory_reserved()]) / 1024 ** 2

# calculate the number of trainable parameters in the model
def get_num_params(model):
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return num_params

print(f"\nGPU memory usage: {get_gpu_memory_usage():.2f} MB")
print(f"Number of trainable parameters: {get_num_params(model):,}")

"""
pridict train/valid/test and get metrics
"""

def evaluate(model, dataloader):
    model.eval()
    total_loss = 0
    total_preds = []
    total_y = []
    for batch in dataloader:
        inputs, y, y_prob = batch
        inputs = {k: v.to(device) for k, v in inputs.items()}
        y = y.to(device)
        with torch.no_grad():
            outputs:SequenceClassifierOutput = model(**inputs, labels = y)
            # loss, logits, hidden_states, attentions = outputs
        # logits = outputs.logits.detach().cpu().numpy()
        # label_ids = y.to("cpu").numpy()
        if y is not None:
            total_loss += outputs.loss.item()
            total_y.append(y.to("cpu").numpy())
        _, pred_y = torch.max(outputs.logits, -1)
        total_preds.append(pred_y.cpu().numpy())
    total_y = np.concatenate(total_y, axis=0) if total_y else None
    total_preds = np.concatenate(total_preds, axis=0)
    avg_loss = total_loss / len(dataloader)
    return total_preds, total_y, avg_loss

evaluate_t0=time.time()
y_preds, y, avg_valid_loss = evaluate(model, valid_dataloader)
valid_f1=f1_score(y, y_preds, average="weighted")
print(f"  Valid weighted F1: {valid_f1:.5f}, Valid evaluation took: {(time.time() - evaluate_t0)/60.0:.2f}m")

test_t0=time.time()
y_preds, y, avg_test_loss = evaluate(model, test_dataloader)
test_f1=f1_score(y, y_preds, average="weighted")
print(f"  Test weighted F1: {test_f1:.5f}, Avg test loss: {avg_test_loss:.6f}, Test evaluation took: {(time.time() - test_t0)/60.0:.2f}m")
print(f"\nGPU memory usage: {get_gpu_memory_usage():.2f} MB")

"""
Test
"""
# avg_test_loss, test_preds, test_y = evaluate(model, test_dataloader)

"""
Examples
"""
def try_examples():
    print("\n\nTry examples:")
    # randomly select 10 samples from test_df
    sample_indices = np.random.randint(0, len(test_df), 10)
    sample_sentences = test_df.iloc[sample_indices][3].tolist()
    sample_y = y_test[sample_indices]


    sample_inputs = tokenizer(
        sample_sentences,
        max_length = MAX_LEN,
        # padding="max_length",
        # padding="do_not_pad",
        padding = True,
        truncation = True,
        return_tensors="pt",
    )
    # print(sample_inputs)
    sample_ids = sample_inputs["input_ids"].view(-1)
    # print(sample_ids.view(-1))
    # 101 cls, 102 sep, 0 pad
    # print(tokenizer.convert_ids_to_tokens(sample_ids))
    print(tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(sample_ids)))

    # classify sentence
    sample_inputs = {k: v.to(device) for k, v in sample_inputs.items()}
    outputs = model(**sample_inputs)
    predictions = outputs.logits.argmax(dim=1)
    print("\n".join([f"p:{p} - y:{y}  {s}" for p, y, s in zip(predictions, sample_y, sample_sentences)]))

# try_examples()

"""
save weights
"""
# model.save_pretrained("/tmp/local-huggingface-models/zhibinlu_vgcn-distilbert-base-uncased-all")
