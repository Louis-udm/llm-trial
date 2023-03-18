import os
from pathlib import Path
import pandas as pd
from sklearn.utils import shuffle
import numpy as np
import random

import transformers as tfr

random.seed(44)
np.random.seed(44)

# set environment variable to use local models
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# specify path to local model files
model_path = (
    "/tmp/local-huggingface-models/zhibinlu_vgcn-distilbert-base-uncased"
)

# load tokenizer and model
tokenizer = tfr.AutoTokenizer.from_pretrained(model_path)
# tokenizer = tfr.VGCNBertTokenizerFast(
#     name_or_path="/tmp/local-huggingface-models/zhibinlu_vgcn-distilbert-base-uncased",
#     vocab_size=30522,
#     model_max_length=1000000000000000019884624838656,
#     is_fast=True,
#     padding_side="right",
#     truncation_side="right",
#     special_tokens={
#         "unk_token": "[UNK]",
#         "sep_token": "[SEP]",
#         "pad_token": "[PAD]",
#         "cls_token": "[CLS]",
#         "mask_token": "[MASK]",
#     },
# )


# example sentence to classify
sentences = ["I really enjoyed this movie!", "My friend doesn't like this movie."]
# [ 101, 1045, 2428, 5632, 2023, 3185,  999,  102]
# sentence = "fdsheibif gjetrsg, suisfdsbewg monfjei, Je suis là! je suis à Montréal! C'est ce que je veux. J'ai vraiment apprécié ce film! C'est ça!"

# tokenize sentence
inputs = tokenizer(sentences[0], return_tensors="pt")
print(inputs)
ids = inputs["input_ids"].view(-1)
print(ids.view(-1))
print(tokenizer.convert_ids_to_tokens(ids))
print(tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(ids)))
