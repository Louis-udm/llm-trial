from safetensors import safe_open
from safetensors.torch import save_file as safe_save_file
from os import path
import torch

# specify path to local model files
model_path = "/tmp/local-huggingface-models/hf-maintainers_distilbert-base-uncased"
# model_path = "/tmp/local-huggingface-models/zhibinlu_vgcn-distilbert-base-uncased"

new_model_path = "/tmp/local-huggingface-models/zhibinlu_vgcn-distilbert-base-uncased"
# new_model_path = "/tmp/local-huggingface-models/zhibinlu_vgcn-distilbert-base-uncased-all"


# safetensors
tensors = {}
with safe_open(path.join(model_path,"model.safetensors"), framework="pt") as f:
    for k in f.keys():
        tensors[k] = f.get_tensor(k)
    metadata=f.metadata()

new_tensors={}
for k,v in tensors.items():
    if k.startswith("distilbert"):
        new_tensors[f"vgcn_bert.{k[11:]}"]=v
    else:
        new_tensors[k]=v

# safe_save_file(new_tensors, path.join(new_model_path,"model.safetensors"), metadata={"format": "pt"})


# torch
# model_path="/tmp/vgcn-bert/model_savings/cola_vb"
state_dict=torch.load(path.join(model_path,"pytorch_model.bin"),map_location=torch.device('cpu'))
# torch.save(new_state_dict, path.join(new_model_path,"pytorch_model.bin"))

print(state_dict)

