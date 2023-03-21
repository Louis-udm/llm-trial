"""
I get a detail PyTorch coding question based on our previous conversation.  It's about the combining of a word-level graph adjacency matrix with Bert-like model. You can assume that I am integrating and refactoring VGCN-Bert model in to Huggingface library using DistilBert. To reduce the computational cost,  there is 2 vocabulary, the base one is for DistilBert tokenizer, we call tok-vocab here, the 2nd vocabulary is a subset from tok-vocab that only include the words when construct the graph to adjacency matrix, we call graph-vocab here, so there will have a mapping btw word indices in tok-vocab and graph-vocab. When a sentence inputted, the model first gets `input_ids` using `DistilBertTokenizer`, then gets the `input_ids_for_graph` according to the mapping.
now assume that the mapping that convert tok-vocab id to graph-vocab id is as following,
```
tok_vocab_id_to_graph_vocab_id_maps={1:10,2:20,3:30, ....}
```
and now I have a input_ids that is under tok-vocab id index, e.g. input_ids=torch.LongTensor([[1,2,3],[3,1,2]]), the shape is (batch, sentence_length), plz help me convert input_ids to graph-vocab id index. 
"""


import torch

tok_vocab_id_to_graph_vocab_id_maps = {10:1, 20:2, 30:3, 40:4, 50:5, 0:-1}

input_ids = torch.LongTensor([[1, 2, 10,20,30, 3,0], [3, 30, 1,10, 2,20,50]])

# create a mapping from tok-vocab id to graph-vocab id
tok_to_graph_mapping = torch.zeros(max(tok_vocab_id_to_graph_vocab_id_maps.keys())+1, dtype=torch.long)-1
for tok_id, graph_id in tok_vocab_id_to_graph_vocab_id_maps.items():
    tok_to_graph_mapping[tok_id] = graph_id

# convert input_ids to graph-vocab id index
input_ids_for_graph = tok_to_graph_mapping[input_ids]
# input_ids_for_graph[input_ids == 0] = -1
# input_ids_mask = input_ids != 0
# input_ids_for_graph *= input_ids_mask
print(input_ids_for_graph)

