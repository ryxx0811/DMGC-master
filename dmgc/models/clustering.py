import torch
import torch.nn as nn

class ClusteringLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,h,u):
        q = 1.0 / (1.0 + torch.sum((h.unsqueeze(1) - u) ** 2, dim=2))
        q = q / torch.sum(q, dim=1, keepdim=True)
        return q
