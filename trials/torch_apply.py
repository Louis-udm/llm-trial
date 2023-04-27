import torch.nn.init as init
import torch.nn as nn
import math

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc1.is_vgcn1=True
        self.fc2 = nn.Linear(20, 5)
        self.fc2.is_vgcn2=True

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x

def init_weights(module):
    if isinstance(module, nn.Linear):
        if hasattr(module, "is_vgcn1") and module.is_vgcn1:
            init.kaiming_uniform_(module.weight, a=math.sqrt(5))
        else:
            init.xavier_uniform_(module.weight)
        if module.bias is not None:
            module.bias.data.fill_(0.01)

model = MyModel()
model.apply(init_weights)