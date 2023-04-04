
import torch
import numpy as np

def print_matrix(m):
    for r in m:
        print(" ".join(["%.1f" % v for v in np.ravel(r)]))
pm=print_matrix

"""
Create the adjacency matrix
"""
g_size=20
def rand_adj():
    adj = torch.sparse_coo_tensor(
        indices=torch.randint(0,g_size,(60,)).view(2,-1),
        values=torch.randint(1,100,(30,))/10,
        size=(g_size, g_size)
    )
    dense_adj=adj.to_dense()
    dense_adj[dense_adj>10]=0
    dense_adj.fill_diagonal_(1.)
    adj=dense_adj.to_sparse_csr()
    return adj

adj=rand_adj()
pm(adj.to_dense())
# adj_matrix = adj_matrix.coalesce()
print(adj)

"""
Create the input_ids tensor
"""
input_ids = torch.tensor([
        [5, 3, 8, 5, 2, 7],
        [0, 2, 0, 3, 2, 2],
        [9, 0, 0, 0, 0, 0]
    ])
x=input_ids

"""
get ground truth sub adj using dense mask
"""
batch_size=x.shape[0]
emb_size=5
gout_size=4
dense_adj=adj.to_dense()
subgraph_mask=torch.zeros((batch_size,)+dense_adj.shape, dtype=torch.float32)
# Compute row and column indices for each x in X
row_idx = x.unsqueeze(-1).repeat(1, 1, subgraph_mask.size(-1))
# Set values in G corresponding to row and column indices to 1
subgraph_mask.scatter_(1, row_idx, 1)
# col_idx = x.unsqueeze(-2).repeat(1, subgraph_mask.size(-2), 1)
col_idx = row_idx.transpose(-1,-2)
subgraph_mask.scatter_(2, col_idx, 1)

one=torch.zeros_like(dense_adj, dtype=torch.float32)
one[:,x[0]]=1
one[x[0],:]=1
assert torch.all(one==subgraph_mask[0])
one=torch.zeros_like(dense_adj, dtype=torch.float32)
one[:,x[1]]=1
one[x[1],:]=1
assert torch.all(one==subgraph_mask[1])

# pm(zadj)
print("--------------")
ground_truth_dense_subadj=dense_adj*subgraph_mask
pm(ground_truth_dense_subadj[0])
print("--------------")
pm(ground_truth_dense_subadj[1])
# c_subadj=cd_subadj.to_sparse_csr()
ground_truth_sparse_subadj=ground_truth_dense_subadj.to_sparse() # coo
print(ground_truth_sparse_subadj)


"""
batch test V*batch_sub_A*W
Batch sparse is only supported by coo in pytorch now
"""
def sub_gcn(V,G,W):
    if G.layout is torch.sparse_coo:
        broadcast_shape = (batch_size,) + W.shape
        GW=torch.bmm(G,W.unsqueeze(0).expand(broadcast_shape))
    else:
        GW=G.matmul(W) # batch,g_size,g_size -> batch, g_size, g_out
    # assert torch.all(GW[0]==G[0].mm(W))
    # assert torch.all(GW[1]==G[1].mm(W))
    assert torch.allclose(GW[0],G[0].mm(W))
    assert torch.allclose(GW[1],G[1].mm(W))
    VGW=V.t().matmul(GW) # (emb, g_size).T.(batch, g_size, g_out) -> batch, emb, g_out
    # assert torch.all(VGW[0]==V.t().mm(GW[0]))
    # assert torch.all(VGW[1]==V.t().mm(GW[1]))
    assert torch.allclose(VGW[0],V.t().mm(GW[0]))
    assert torch.allclose(VGW[1],V.t().mm(GW[1]))
    return VGW.transpose(1,2) # batch, g_out, emb

def sub_gcn_use_fc(V,G,W):
# def sub_gcn(V,G,W):
    fc=torch.nn.Linear(g_size, gout_size, bias=False)
    fc.weight.data=W.t()
    # torch.nn.fc does not support sparse
    GW=fc(G) # batch,g_size,g_size -> batch, g_size, g_out
    # assert torch.all(GW[0]==G[0].mm(W))
    # assert torch.all(GW[1]==G[1].mm(W))
    assert torch.allclose(GW[0],G[0].mm(W))
    assert torch.allclose(GW[1],G[1].mm(W))
    VGW=V.t().matmul(GW) # emb, g_size * batch, g_out, w_size -> batch, emb, g_out
    # assert torch.all(VGW[0]==V.t().mm(GW[0]))
    # assert torch.all(VGW[1]==V.t().mm(GW[1]))
    assert torch.allclose(VGW[0],V.t().mm(GW[0]))
    assert torch.allclose(VGW[1],V.t().mm(GW[1]))
    return VGW.transpose(1,2) # batch, g_out, emb

V=torch.rand((g_size, emb_size))
W=torch.rand((g_size, gout_size))
res1=sub_gcn(V,ground_truth_dense_subadj,W) # batch, g_out, emb
res2=sub_gcn(V,ground_truth_sparse_subadj,W) # batch, g_out, emb
assert torch.allclose(res1,res2)

"""
get batch_masks form input x according to adj_coo
"""
batch_size=x.shape[0]
adj_coo=adj.to_sparse_coo()
batch_masks=torch.any(torch.any((adj_coo.indices().view(-1)==x.unsqueeze(-1)).view(batch_size,x.shape[1],2,-1), dim=1),dim=1)
# mask is better than mask1
mask1=torch.any(torch.any((adj_coo.indices().view(-1)==x.unsqueeze(-1)), dim=1).view(batch_size,2,-1), dim=1)
assert torch.all(batch_masks==mask1)

"""
test one by one (one mask) for get sub_sparse_adj from sparse adj
"""
one_sub_sparse_adj = torch.sparse_coo_tensor(
    indices=adj_coo.indices()[:, batch_masks[0]],
    values=adj_coo.values()[batch_masks[0]],
    size=(g_size, g_size),
)
assert torch.all(ground_truth_sparse_subadj[0].to_dense()==one_sub_sparse_adj.to_dense())

one_sub_sparse_adj = torch.sparse_coo_tensor(
    indices=adj_coo.indices()[:, batch_masks[1]],
    values=adj_coo.values()[batch_masks[1]],
    size=(g_size, g_size),
)
assert torch.all(ground_truth_sparse_subadj[1].to_dense()==one_sub_sparse_adj.to_dense())

"""
Finally, sparse graph to batch sparse graph
"""
def get_batch_sub_adj_coo(adj_coo, batch_masks):

    batch_size = batch_masks.size(0)
    nnz_len=len(adj_coo.values())

    # Convert adj_coo.values to a tensor with shape (batch_size, adj_coo.nnz)
    values = adj_coo.values().unsqueeze(0).repeat(batch_size, 1)

    # # Apply the mask to the values tensor
    # values *= batch_masks.float()
    # # Create a batch_indices tensor with shape (batch_size, adj_coo.nnz)
    # batch_positions = torch.arange(batch_size).unsqueeze(1).repeat(1, nnz_len)
    # # Create a row_indices tensor with shape (batch_size, adj_coo.nnz)
    # row_indices = adj_coo.indices()[0].unsqueeze(0).repeat(batch_size, 1)
    # # Apply the mask to the row_indices tensor
    # row_indices *= batch_masks.long()
    # # Create a row_indices tensor with shape (batch_size, adj_coo.nnz)
    # col_indices = adj_coo.indices()[1].unsqueeze(0).repeat(batch_size, 1)
    # # Apply the mask to the row_indices tensor
    # col_indices *= batch_masks.long()
    # indices=torch.stack([batch_positions.view(-1),row_indices.view(-1),col_indices.view(-1)])

    values=values.view(-1)[batch_masks.view(-1)]

    batch_positions = torch.arange(batch_size).unsqueeze(1).repeat(1, nnz_len)
    indices=torch.cat([batch_positions.view(1,-1), adj_coo.indices().repeat(1,batch_size)],dim=0)
    indices=indices[batch_masks.view(-1).expand(3,-1)].view(3,-1)


    # Create the batch_sub_adj_coo tensor
    batch_sub_adj_coo = torch.sparse_coo_tensor(
        indices=indices,
        values=values.view(-1),
        size=(batch_size, adj_coo.size(0), adj_coo.size(1)),
        # dtype=adj_coo.dtype,
        # device=adj_coo.device
    )

    return batch_sub_adj_coo.coalesce()

sparse_subadj=get_batch_sub_adj_coo(adj_coo,batch_masks)

assert torch.allclose(sparse_subadj.to_dense(),ground_truth_sparse_subadj.to_dense())
res3=sub_gcn(V,sparse_subadj,W)
assert torch.allclose(res1,res3)

pm(res3[0])


