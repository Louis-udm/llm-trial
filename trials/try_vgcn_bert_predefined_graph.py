import os
from pathlib import Path
import pandas as pd
from sklearn.utils import shuffle
import numpy as np
import random
import dill
import torch

import transformers as tfr

random.seed(44)
np.random.seed(44)
torch.manual_seed(44)

cuda_yes = torch.cuda.is_available()
if cuda_yes:
    torch.cuda.manual_seed_all(44)
device = torch.device("cuda:0" if cuda_yes else "cpu")


def print_matrix(m):
    for r in m:
        print(" ".join(["%.1f" % v for v in np.ravel(r)]))

# set environment variable to use local models
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# specify path to local model files
model_path = (
    "/tmp/local-huggingface-models/zhibinlu_vgcn-distilbert-base-uncased"
)

# load tokenizer and model
tokenizer = tfr.AutoTokenizer.from_pretrained(model_path)

"""
build/Load wgraph
"""

from transformers.models.vgcn_bert.modeling_graph import WordGraph,_normalize_adj

words_relations = [
    ("like", "enjoy", 0.7),
    ("love", "enjoy", 0.8),
    ("hate", "like", -0.4),
    ("fantastic", "great", 0.4),
    ("waste", "like", -0.6),
]
wgraph = WordGraph(words_relations, tokenizer)
print(wgraph.vocab_indices)
print_matrix(wgraph.adjacency_matrix.todense())

# example sentence to classify
sentences = ["I really enjoyed this movie!", "I love this movie!", "I like this movie!",
             "I hate this movie!", "A boring movie.", "I do not like this movie!",
             "It is a waste of time.", "Do not waste your time on this movie.",
             "I do not enjoy this movie!", "I do not love this movie!", "I don't like this movie!"]
# sentence = "fdsheibif gjetrsg, suisfdsbewg monfjei, Je suis là! je suis à Montréal! C'est ce que je veux. J'ai vraiment apprécié ce film! C'est ça!"


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
print("Prediction:")
print("\n".join([f"{labels[p]}, {s}" for p,s in zip(predictions, sentences)]))

"""
save weights
"""
# model.save_pretrained("/tmp/local-huggingface-models/zhibinlu_vgcn-distilbert-base-uncased-all")
