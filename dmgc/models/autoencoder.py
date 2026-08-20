import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphAutoEncoder(nn.Module):
    def __init__(self,dims):
        super().__init__()
        n_stack = len(dims)-1
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for i in range(n_stack):
            linear = nn.Linear(in_features=dims[i],out_features=dims[i+1])
            nn.init.xavier_uniform_(linear.weight)
            self.encoders.append(linear)
        for i in range(n_stack-1,-1,-1):
            linear = nn.Linear(in_features=dims[i+1],out_features=dims[i])
            nn.init.xavier_uniform_(linear.weight)
            self.decoders.append(linear)

    def forward(self,x):
        h = x
        for encoder in self.encoders:
            h = encoder(h)
            h = F.relu(h)
        emb = h
        a = h
        for decoder in self.decoders:
            a = decoder(a)
            a = F.relu(a)
        return emb,a




