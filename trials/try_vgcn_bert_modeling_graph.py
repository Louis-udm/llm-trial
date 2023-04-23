import transformers as tfr
import transformers.models.vgcn_bert.modeling_graph as tfrvgcn


if __name__ == "__main__":
    import os


    def print_matrix(m):
        for r in m:
            print(" ".join(["%.1f" % v for v in np.ravel(r)]))

    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    model_path = "/tmp/local-huggingface-models/hf-maintainers_distilbert-base-uncased"

    # DistilBertTokenizerFast
    tokenizer = tfr.AutoTokenizer.from_pretrained(model_path)

    words_relations = [
        ("I", "you", 0.3),
        ("here", "there", 0.7),
        ("city", "montreal", 0.8),
        ("comeabc", "gobefbef", 0.2),
    ]
    wgraph = tfrvgcn.WordGraph(words_relations, tokenizer)
    print(len(wgraph.vocab_indices))
    # print(wgraph.tokenizer_id_to_wgraph_id_array)
    print_matrix(wgraph.adjacency_matrix.todense())

    # texts = [" I am here", "He is here", "here i am, gobefbef"]
    texts = [" I am here!", "He is here", "You are also here, gobefbef!", "What is interpribility"]
    wgraph = tfrvgcn.WordGraph(texts, tokenizer, window_size=4)

    print(len(wgraph.vocab_indices))
    # print(wgraph.tokenizer_id_to_wgraph_id_array)
    print_matrix(wgraph.adjacency_matrix.todense())
    print()
    norm_adj = tfrvgcn._normalize_adj(wgraph.adjacency_matrix)
    print_matrix(norm_adj.todense())

    # print(vocab_indices[vocab[3]])
    print("---end---")
