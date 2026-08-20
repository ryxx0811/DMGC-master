import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossGraphLayer(nn.Module):
    def __init__(self,network_id,n_networks,center_dim,output_dim,n_clusters):
        super().__init__()
        self.network_id = network_id
        self.n_networks = n_networks
        self.n_clusters = n_clusters

        self.w = nn.Parameter(torch.empty(n_networks*center_dim,output_dim))
        nn.init.xavier_uniform_(self.w)


    def forward(self,Cs,zs,):

        reshaped_centers = []
        for i in range(self.n_networks):
            if i == self.network_id:

                reshaped_centers.append(zs[i])
            else:
                reshaped_centers.append(torch.matmul(Cs[i], zs[i]))
        concat_centers = torch.concat(reshaped_centers, dim=-1)
        u = F.relu(torch.matmul(concat_centers,self.w))
        return u


