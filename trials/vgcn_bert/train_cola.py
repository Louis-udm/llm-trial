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
from torch.optim import SparseAdam
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
        # inputs = self.tokenizer.encode_plus(
        #     text=sentence,
        #     text_pair=None,
        #     add_special_tokens=True,
        #     max_length=self.max_len,
        #     # pad_to_max_length=True,
        #     padding="max_length",
        #     # padding="longest",
        #     # return_token_type_ids=True,
        #     truncation=True,
        #     return_tensors="pt",
        # )
        inputs = self.tokenizer(
            text=sentence,
            text_pair=None,
            padding="max_length", 
            max_length=self.max_len, 
            truncation=True,
            return_tensors="pt",
        )
        inputs={k:v[0] for k,v in inputs.items()}
        return inputs,y,y_prob
    
MAX_LEN = 512-16
TRAIN_BATCH_SIZE = 2
VALID_BATCH_SIZE = 32
TOTAL_EPOCH = 9
LEARNING_RATE = 5e-5 #1e-05 # 2e-5, 5e-5, old 8e-6
DROPOUT_RATE = 0.2  # 0.5 # Dropout rate (1 - keep probability).
L2_DECAY = 0.001
DO_LOWER_CASE = True

train_dataloader = DataLoader(
    dataset=ColaDataset(
        train_valid_df[3][:train_size], 
        y_train_valid[:train_size], 
        y_prob_train_valid[:train_size], 
        tokenizer,
        MAX_LEN
    ), batch_size=TRAIN_BATCH_SIZE, shuffle=False)
valid_dataloader = DataLoader(
    dataset=ColaDataset(
        train_valid_df[3][train_size:], 
        y_train_valid[train_size:], 
        y_prob_train_valid[train_size:], 
        tokenizer,
        MAX_LEN
    ), batch_size=VALID_BATCH_SIZE, shuffle=False)
test_dataloader = DataLoader(
    dataset=ColaDataset(
        test_df[3], 
        y_test, 
        y_prob_test, 
        tokenizer,
        MAX_LEN
    ), batch_size=VALID_BATCH_SIZE, shuffle=False)

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
    total_preds = []
    total_y = []
    for batch in dataloader:
        inputs, y, y_prob = batch
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs:SequenceClassifierOutput = model(**inputs, labels=y)
            # loss, logits, hidden_states, attentions = outputs
        # logits = outputs.logits.detach().cpu().numpy()
        # label_ids = y.to("cpu").numpy()
        total_loss += outputs.loss.item()
        _, pred_y = torch.max(outputs.logits, -1)
        total_preds.append(pred_y.tolist())
        total_y.append(y.tolist())
    avg_loss = total_loss / len(dataloader)
    total_preds = np.concatenate(total_preds, axis=0)
    total_y = np.concatenate(total_y, axis=0)
    return avg_loss, total_preds, total_y


def train(model, train_dataloader, valid_dataloader, test_dataloader, optimizer):
    train_start = time.time()
    all_loss_list = {"train": [], "valid": [], "test": []}
    all_f1_list = {"train": [], "valid": [], "test": []}
    len_train_ds = len(train_dataloader)
    for epoch in range(TOTAL_EPOCH):
        model.train()
        total_loss = 0
        optimizer.zero_grad()
        for step, batch in enumerate(train_dataloader):
            inputs, y_train, y_prob_train = batch
            inputs = {k: v.to(device) for k, v in inputs.items()}
            y_train=y_train.to(device)
            outputs:SequenceClassifierOutput = model(**inputs, labels=y_train)
            # loss, logits, hidden_states, attentions = outputs
            outputs.loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += outputs.loss.item()
            if step % 40 == 0:
                print(
                    "Epoch:{}-{}/{}, Train Loss: {}, Cumulated time: {}m ".format(
                        epoch,
                        step,
                        len_train_ds,
                        outputs.loss.item(),
                        (time.time() - train_start) / 60.0,
                    )
                )
        avg_train_loss = total_loss / len_train_ds
        evaluate_t0=time.time()
        avg_valid_loss, y_preds, y = evaluate(model, valid_dataloader)
        valid_f1=f1_score(y,y_preds,average="weighted")
        print("--------------------------------------------------------------")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Average valid loss: {0:.2f}".format(avg_valid_loss))
        print("  Average valid F1: {0:.2f}".format(valid_f1))
        print("  Valid evaluation took: {:}".format(format_time(time.time() - evaluate_t0)))

        test_t0=time.time()
        avg_test_loss, y_preds, y = evaluate(model, test_dataloader)
        test_f1=f1_score(y,y_preds,average="weighted")
        print("  Average test loss: {0:.2f}".format(avg_test_loss))
        print("  Average test F1: {0:.2f}".format(test_f1))
        print("  Test evaluation took: {:}".format(format_time(time.time() - test_t0)))
        
        all_loss_list["train"].append(avg_train_loss)
        all_loss_list["valid"].append(avg_valid_loss)
        all_loss_list["test"].append(avg_test_loss)
        all_f1_list["valid"].append(valid_f1)
        all_f1_list["test"].append(test_f1)

    best_valid_f1 = max(all_f1_list["valid"])
    idx_best_valid_f1 = all_f1_list["valid"].index(best_valid_f1)
    print(f"\n**Optimization Finished!,Total spend: {format_time(time.time() - train_start)}")
    print("**Best Valid weighted F1: %.3f at %d epoch."% (100 * best_valid_f1, idx_best_valid_f1))
    print("**Test weighted F1 when valid best: %.3f" % (100 * all_f1_list["test"][idx_best_valid_f1]))



# optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
optimizer = SparseAdam(model.parameters(), lr=LEARNING_RATE, eps=1e-8)
train(model, train_dataloader, valid_dataloader, test_dataloader, optimizer)

"""
Test
"""
# avg_test_loss, test_preds, test_y = evaluate(model, test_dataloader)

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
