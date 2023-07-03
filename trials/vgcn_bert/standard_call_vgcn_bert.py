import transformers as tfr
from transformers.models.vgcn_bert.modeling_graph import WordGraph
from transformers.models.vgcn_bert.modeling_vgcn_bert import VGCNBertModel

tokenizer = tfr.AutoTokenizer.from_pretrained(
    "zhibinlu/vgcn-distilbert-base-uncased"
)
# 1st method: Build graph using NPMI statistical method from training corpus
# wgraph = WordGraph(rows=train_valid_df["text"], tokenizer=tokenizer)
# 2nd method: Build graph from pre-defined entity relationship tuple with weight
entity_relations = [
    ("dog", "labrador", 0.6),
    ("cat", "garfield", 0.7),
    ("city", "montreal", 0.8),
    ("weather", "rain", 0.3),
]
wgraph = WordGraph(rows=entity_relations, tokenizer=tokenizer)

model = VGCNBertModel.from_pretrained(
    "zhibinlu/vgcn-distilbert-base-uncased",
    wgraphs=[wgraph.to_torch_sparse()],
    wgraph_id_to_tokenizer_id_maps=[wgraph.wgraph_id_to_tokenizer_id_map],
)
text = "Replace me by any text you'd like."
encoded_input = tokenizer(text, return_tensors="pt")
output = model(**encoded_input)
print(output.last_hidden_state.shape)
