import os
from pathlib import Path
import pandas as pd
from sklearn.utils import shuffle
import numpy as np
import random

# tokenizer = AutoTokenizer.from_pretrained("nreimers/MiniLMv2-L6-H384-distilled-from-RoBERTa-Large")
# model = AutoModel.from_pretrained("nreimers/MiniLMv2-L6-H384-distilled-from-RoBERTa-Large")
import transformers as tfr

random.seed(44)
np.random.seed(44)

valid_data_taux = 0.05
test_data_taux = 0.10
# set environment variable to use local models
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# specify path to local model files
model_path = "/tmp/local-huggingface-models/zhibinlu_vgcn-distilbert-base-uncased"

# load tokenizer and model
# tokenizer = tfr.VGCNBertTokenizerFast(
# tokenizer = tfr.AutoTokenizer.from_pretrained(model_path)
tokenizer = tfr.DistilBertTokenizerFast.from_pretrained(model_path)


ds_path="/home/zhibin/Projects/VGCN-BERT/data/CoLa"
# train_valid_df=pd.read_csv(os.path.join(ds_path,"ds_train_valid.csv"), header=0, index_col=0)
# test_df=pd.read_csv(os.path.join(ds_path,"ds_test.csv"))

label2idx = {"0": 0, "1": 1}
idx2label = {0: "0", 1: "1"}
train_valid_df = pd.read_csv(
    os.path.join(ds_path,"train.tsv"), encoding="utf-8", header=None, sep="\t"
)
train_valid_df = shuffle(train_valid_df)
# use dev set as test set, because we can not get the ground true label of the real test set.
test_df = pd.read_csv(
    os.path.join(ds_path,"dev.tsv"), encoding="utf-8", header=None, sep="\t"
)

train_valid_size = train_valid_df[1].count()
valid_size = int(train_valid_size * valid_data_taux)
train_size = train_valid_size - valid_size
test_size = test_df[1].count()
print(
    "CoLA train_valid Total:", train_valid_size, "test Total:", test_size
)
# df = pd.concat((train_valid_df, test_df))
# corpus = df[3]
# y = df[1].values  # y.as_matrix()
# confidence
# y_prob = np.eye(len(y), len(label2idx))[y]
# corpus_size = len(y)

y_train_valid=train_valid_df[1].values
y_test=test_df[1].values
y_prob_train_valid=np.eye(len(y_train_valid), len(label2idx))[y_train_valid]
y_prob_test=np.eye(len(y_test), len(label2idx))[y_test]

#----------------------------------
from transformers.models.vgcn_bert.modeling_graph import WordGraph

import random
wgraph=WordGraph(rows=train_valid_df[3], tokenizer=tokenizer)
vocab_adj,vocab,vocab_indices, vocab_wgraph2tokenizer_id_map=wgraph.adjacency_matrix,wgraph.vocab,wgraph.vocab_indices,wgraph.vocab_wgraph2tokenizer_id_map

print(
    "  Zero ratio(?>66%%) of the graph matrix: %.8f"
    % (100 * (1 - vocab_adj.count_nonzero() / (vocab_adj.shape[0] * vocab_adj.shape[1])))
)

torch_graph=wgraph.to_torch_sparse()

# DistilBertForSequenceClassification
model = tfr.AutoModelForSequenceClassification.from_pretrained(model_path,vgcn_graphs=[torch_graph],vgcn_vocab2tokenizer_map=vocab_wgraph2tokenizer_id_map)
# model = tfr.AutoModel.from_pretrained(model_path)

# example sentence to classify
sentence = "I really enjoyed this movie!"
# sentence = "fdsheibif gjetrsg, suisfdsbewg monfjei, Je suis là! je suis à Montréal! C'est ce que je veux. J'ai vraiment apprécié ce film! C'est ça!"

# tokenize sentence
# DistilBertTokenizerFast
inputs = tokenizer(sentence, return_tensors="pt")
print(inputs)
ids=inputs["input_ids"].view(-1)
print(ids.view(-1))
print(tokenizer.convert_ids_to_tokens(ids))
print(tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(ids)))

# # classify sentence
outputs = model(**inputs)
predictions = outputs.logits.argmax(dim=1)

# # print predicted label
labels = ["negative", "positive"]
print("Prediction:", labels[predictions])
