import os
from pathlib import Path

# tokenizer = AutoTokenizer.from_pretrained("nreimers/MiniLMv2-L6-H384-distilled-from-RoBERTa-Large")
# model = AutoModel.from_pretrained("nreimers/MiniLMv2-L6-H384-distilled-from-RoBERTa-Large")
import transformers as tfr

# set environment variable to use local models
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# specify path to local model files
model_path = "/tmp/local-huggingface-models/hf-maintainers_bert-base-uncased"

# load tokenizer and model
tokenizer = tfr.AutoTokenizer.from_pretrained(model_path)
# tokenizer = tfr.RobertaTokenizerFast(
#     name_or_path="models/MiniLMv2-L6-H384-distilled-from-RoBERTa-Large",
#     vocab_size=50265,
#     model_max_length=512,
#     is_fast=True,
#     padding_side="right",
#     truncation_side="right",
#     special_tokens={
#         "bos_token": "<s>",
#         "eos_token": "</s>",
#         "unk_token": "<unk>",
#         "sep_token": "</s>",
#         "pad_token": "<pad>",
#         "cls_token": "<s>",
#         "mask_token": tfr.AddedToken(
#             "<mask>",
#             rstrip=False,
#             lstrip=True,
#             single_word=False,
#             normalized=False,
#         ),
#     },
# )

# RobertaForMaskedLM
model = tfr.AutoModelForSequenceClassification.from_pretrained(model_path)
# model = tfr.AutoModel.from_pretrained(model_path)

# example sentence to classify
sentence = "I really enjoyed this movie!"
sentence = "fdsheibif gjetrsg, suisfdsbewg monfjei, Je suis là! je suis à Montréal! C'est ce que je veux. J'ai vraiment apprécié ce film! C'est ça!"

# tokenize sentence
inputs = tokenizer(sentence, return_tensors="pt")
print(inputs)
ids=inputs["input_ids"].view(-1)
print(ids.view(-1))
print(tokenizer.convert_ids_to_tokens(ids))
print(tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(ids)))

# # classify sentence
# outputs = model(**inputs)
# predictions = outputs.logits.argmax(dim=1)

# # print predicted label
# labels = ["negative", "positive"]
# print("Prediction:", labels[predictions])
