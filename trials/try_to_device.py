import torch
import torch.nn as nn
from typing import List

class VGCN(nn.Module):
    def __init__(self, t_list: List[torch.Tensor]):
        super().__init__()
        self.fc=nn.Linear(768, 768)
        self.fc_list=[nn.Linear(768, 768), nn.Linear(768, 768), nn.Linear(768, 768)]
        self.fc_list=nn.ModuleList(self.fc_list)
        self.t_list = t_list
    
    def forward(self, x):
        x=self.fc(x)
        for fc in self.fc_list:
            x=fc(x)
        for t in self.t_list:
            x=torch.matmul(t, x)
        return x

    def to(self, device):
        # move model to device
        super().to(device)
        # move each tensor in t_list to device
        self.t_list=[t.to(device) for t in self.t_list]
        return self

class GNN(nn.Module):
    def __init__(self, t_list: List[torch.Tensor]):
        super().__init__()
        self.gcn=VGCN(t_list=t_list)
        self.fc=nn.Linear(768, 768)
    
    def forward(self,x):
        x= self.gcn(x)
        return self.fc(x)

t_list=[torch.rand(768, 768), torch.rand(768, 768), torch.rand(768, 768)]
gnn=GNN(t_list=t_list)
x=torch.rand(768)
x=x.to(device="cuda:0")
gnn.to(device="cuda:0")
# gnn.gcn.to(device="cuda:0")
res=gnn(x)
print(res.shape)
