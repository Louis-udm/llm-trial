import os
from pathlib import Path
import pandas as pd
from sklearn.utils import shuffle
import numpy as np
import random
import dill
import torch

# tokenizer = AutoTokenizer.from_pretrained("nreimers/MiniLMv2-L6-H384-distilled-from-RoBERTa-Large")
# model = AutoModel.from_pretrained("nreimers/MiniLMv2-L6-H384-distilled-from-RoBERTa-Large")
import transformers as tfr

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

ds_path = "../VGCN-BERT/data/CoLA"
# train_valid_df=pd.read_csv(os.path.join(ds_path,"ds_train_valid.csv"), header=0, index_col=0)
# test_df=pd.read_csv(os.path.join(ds_path,"ds_test.csv"))

valid_data_taux = 0.05
test_data_taux = 0.10

label2idx = {"0": 0, "1": 1}
idx2label = {0: "0", 1: "1"}
train_valid_df = pd.read_csv(
    os.path.join(ds_path, "train.tsv"), encoding="utf-8", header=None, sep="\t"
)
train_valid_df = shuffle(train_valid_df)
# use dev set as test set, because we can not get the ground true label of the real test set.
test_df = pd.read_csv(
    os.path.join(ds_path, "dev.tsv"), encoding="utf-8", header=None, sep="\t"
)

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

from transformers.models.vgcn_bert.modeling_graph import WordGraph,_normalize_adj

cola_wgraph_path = "/tmp/vgcn-bert/cola_wgraph.pkl"
# wgraph=WordGraph(rows=train_valid_df[3], tokenizer=tokenizer)
# with open(cola_wgraph_path, "wb") as f:
#     dill.dump(wgraph,f)
with open(cola_wgraph_path, "rb") as f:
    wgraph = dill.load(f)

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

def compare_normalizations(adj):
    # this is for compare the original adj and zero-padding adj, see if they are the same
    # the result is the same
    if adj[0,:].sum()==1:
        adj1=adj.copy()
        nadj1=_normalize_adj(adj1)
        nadj2=_normalize_adj(adj.copy()[1:,1:])
        adj3=adj.copy()
        adj3[0,0]=0
        nadj3=_normalize_adj(adj3)

        # adding zero padding(add 0 row/col) for adj_matrix does not affect normalization
        print(nadj2.sum()==nadj3.sum())
        n1=nadj1.tocoo().copy()
        n2=nadj2.tocoo().copy()
        n3=nadj3.tocoo().copy()
        print((n2.data==n3.data).all())
    else:
        print("adj already has zero padding.")

compare_normalizations(wgraph.adjacency_matrix)


"""
Init Model
"""
torch_graph = wgraph.to_torch_sparse()

model = tfr.AutoModelForSequenceClassification.from_pretrained(
    model_path,
    [torch_graph],
    [wgraph.wgraph_id_to_tokenizer_id_map],
)
model.to(device)
model.vgcn_bert.embeddings.vgcn.set_transparent_parameters()
# model = tfr.AutoModel.from_pretrained(model_path)

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
