import torch
import torch.nn as nn
from .autoencoder import GraphAutoEncoder
from .clustering import ClusteringLayer
from .cross_graph import CrossGraphLayer

#shape= [13,664]

class DMGC(nn.Module):
    def __init__(self,n_networks,dims,center_dim,n_clusters):
        super().__init__()
        self.n_networks = n_networks

        self.zs = nn.ParameterList()
        self.h = [None] * n_networks
        self.a  = [None] * n_networks
        self.u = [None] * n_networks
        self.Cs = [None] * n_networks
        self.qs = [None] * n_networks
        #self.vl = [None] * n_networks #gnn

        self.n_clusters = n_clusters

        for i in range(n_networks):
            z = nn.Parameter(torch.empty(n_clusters[i], center_dim))
            nn.init.xavier_uniform_(z)
            self.zs.append(z)


        self.autoencoders = nn.ModuleList([GraphAutoEncoder(dims=dims[i]) for i in range(self.n_networks)]) #fnn
        #self.vgae = nn.ModuleList([VariationalGraphAutoEncoder(dims=dims[i]) for i in range(self.n_networks) ]) #gnn
        self.crosslayers = nn.ModuleList([
            CrossGraphLayer(
                network_id=i,
                n_networks=n_networks,
                center_dim=center_dim,
                output_dim=dims[i][-1],
                n_clusters=n_clusters[i],
            )
            for i in range(n_networks)
        ])
        self.clusteringlayers = nn.ModuleList([
            ClusteringLayer() for _ in range(n_networks)
        ]
        )

    def compute_Cs(self):
        Cs = [None] * self.n_networks
        for i in range(self.n_networks):
            Cs[i] = [None] * self.n_networks

            Cs[i][i] = torch.tensor(0.0)
            mu_i = self.zs[i]
            for j in range(self.n_networks):
                mu_j = self.zs[j]
                if i == j:
                    continue

                temp = torch.tile(mu_i, (self.n_clusters[j], 1))
                tile_vec_i = torch.reshape(temp, (self.n_clusters[j], mu_i.shape[0], mu_i.shape[1]))
                re_tile_vec_i = torch.transpose(tile_vec_i, 0, 1)

                temp = torch.tile(mu_j, (self.n_clusters[i], 1))
                tile_vec_j = torch.reshape(temp, [self.n_clusters[i], mu_j.shape[0], mu_i.shape[1]])

                diff = re_tile_vec_i - tile_vec_j
                reduce_sum_diff = torch.sum(diff.pow(2), dim=2)

                t_sim = 1.0 / (1 + reduce_sum_diff)
                logtis = t_sim
                Cs[i][j] = logtis / (torch.sum(logtis, dim=0))
        return Cs

    def forward(self,x):
        self.Cs=self.compute_Cs()
        for i in range(self.n_networks):

            self.h[i], self.a[i] = self.autoencoders[i].forward(x[i]) #fnn
            #self.h[i], self.a[i],self.vl[i]= self.vgae[i].forward(x=torch.eye(shape[i]) ,A=x[i],W=w[i]) #gnn
            self.u[i] = self.crosslayers[i].forward(self.Cs[i],self.zs)
            self.qs[i] = self.clusteringlayers[i].forward(self.h[i],self.u[i])


        return self.h,self.a,self.zs,self.u,self.Cs,self.qs #fnn

        #return self.h,self.a,self.zs,self.u,self.Cs,self.qs,self.vl #gnn

