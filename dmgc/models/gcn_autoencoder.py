import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GAE, GATConv, Linear

class GraphAutoEncoder(nn.Module):
    def __init__(self,dims):
        super().__init__()
        n_stack = len(dims)-1
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for i in range(n_stack):
            conv = GCNConv(in_channels=dims[i], out_channels=dims[i+1])
            self.encoders.append(conv)
        for i in range(n_stack-1,-1,-1):
            conv = GCNConv(in_channels=dims[i+1],out_channels=dims[i])
            self.decoders.append(conv)

    def forward(self,x,edge_index,edge_weight):
        h = x
        for encoder in self.encoders:
            h = encoder(h,edge_index,edge_weight)
            h = F.relu(h)
        emb = h
        a = h
        for decoder in self.decoders:
            a = decoder(a,edge_index,edge_weight)
            a = F.relu(a)
        return emb,a
