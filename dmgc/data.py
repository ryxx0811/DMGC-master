import networkx as nx
import os

import torch

from dmgc.utils import get_edges, logging_info


class GraphDataLoader:
    def __init__(self, task, n_networks, n_clusters, dims, center_dim, beta, output_dir='results'):
        self.task = task
        self.out_path = os.path.join(output_dir, self.task)
        self.n_networks = n_networks
        self.n_clusters = n_clusters
        self.dims = dims
        self.center_dim = center_dim
        self.beta = beta
        self.xs, self.input,self.cross = self.load_data()

        logging_info(self.out_path)

    def load_data(self):
        if self.task == 'spatial':
            return self.load_spatial()
        elif self.task == 'heart':
            return self.load_heart()
        elif self.task == 'single_cell':
            return self.load_single_cell()
        else:
            raise ValueError(f"Invalid task: {self.task}. Expected 'spatial' or 'single_cell'.")
    def load_heart(self):
        chaffin_path = 'reheatHeart/data/chaffin_adj_topk.pt'
        kuppe_path = 'reheatHeart/data/kuppe_adj_topk.pt'
        koenig_path = 'reheatHeart/data/koenig_adj_topk.pt'
        reichart_path = 'reheatHeart/data/reichart_adj_topk.pt'
        simonson_path = 'reheatHeart/data/simonson_adj_topk.pt'
        cross_path = 'reheatHeart/data/cross.pt'

        A = torch.load(chaffin_path)
        chaffin_adj_topk = torch.tensor(A, dtype=torch.float32)
        A = torch.load(kuppe_path)
        kuppe_adj_topk = torch.tensor(A, dtype=torch.float32)
        A = torch.load(koenig_path)
        koenig_adj_topk = torch.tensor(A, dtype=torch.float32)
        A = torch.load(reichart_path)
        reichart_adj_topk = torch.tensor(A, dtype=torch.float32)
        A = torch.load(simonson_path)
        simonson_adj_topk = torch.tensor(A, dtype=torch.float32)
        A = torch.load(cross_path)
        cross = torch.tensor(A, dtype=torch.float32)

        chaffin_path = 'reheatHeart/data/chaffin_adj.pt'
        kuppe_path = 'reheatHeart/data/kuppe_adj.pt'
        koenig_path = 'reheatHeart/data/koenig_adj.pt'
        reichart_path = 'reheatHeart/data/reichart_adj.pt'
        simonson_path = 'reheatHeart/data/simonson_adj.pt'


        A = torch.load(chaffin_path)
        chaffin_adj = torch.tensor(A, dtype=torch.float32)
        A = torch.load(kuppe_path)
        kuppe_adj = torch.tensor(A, dtype=torch.float32)
        A = torch.load(koenig_path)
        koenig_adj = torch.tensor(A, dtype=torch.float32)
        A = torch.load(reichart_path)
        reichart_adj = torch.tensor(A, dtype=torch.float32)
        A = torch.load(simonson_path)
        simonson_adj = torch.tensor(A, dtype=torch.float32)

        return [chaffin_adj_topk, kuppe_adj_topk, koenig_adj_topk, reichart_adj_topk, simonson_adj_topk],[chaffin_adj_topk, kuppe_adj_topk, koenig_adj_topk, reichart_adj_topk, simonson_adj_topk], cross



    def load_spatial(self):
        cts = ['GCBC','NBC_MBC','FDC','epithelial','CD4_T','myeloid']
        #cts = ['GCBC','NBC_MBC','PC','CD4_T']
        intra_path = 'data/spatial/graph/intra/'
        intras = []
        for ct in cts:
            pt_path = os.path.join(intra_path, f'{ct}.pt')
            A = torch.load(pt_path,weights_only = False)
            intras.append(torch.tensor(A).float())

        inter_path = 'data/spatial/graph/inter/'
        inters = []
        for i in range(len(cts)):
            for j in range(len(cts)):
                if i < j:
                    pt_path = os.path.join(inter_path, f'{cts[i]}_{cts[j]}.pt')
                    A = torch.load(pt_path,weights_only = False)
                    inters.append(A.float())

        return intras, intras, inters

    def load_single_cell(self):
        lung_path = 'data/single/lung_topk.pt'
        kidney_path = 'data/single/kidney_topk.pt'
        heart_path = 'data/single/heart_topk.pt'
        liver_path = 'data/single/liver_topk.pt'
        cross_path = 'data/single/cross.pt'

        A = torch.load(lung_path)
        lung_adj_topk = torch.tensor(A, dtype=torch.float32)
        A = torch.load(kidney_path)
        kidney_adj_topk = torch.tensor(A, dtype=torch.float32)
        A = torch.load(heart_path)
        heart_adj_topk = torch.tensor(A, dtype=torch.float32)
        A = torch.load(liver_path)
        liver_adj_topk = torch.tensor(A, dtype=torch.float32)
        A = torch.load(cross_path)
        lung_kidney_heart_liver_adj = torch.tensor(A, dtype=torch.float32)

        lung_path = 'data/single/lung.pt'
        kidney_path = 'data/single/kidney.pt'
        heart_path = 'data/single/heart.pt'
        liver_path = 'data/single/liver.pt'


        A = torch.load(lung_path)
        lung_adj = torch.tensor(A, dtype=torch.float32)
        A = torch.load(kidney_path)
        kidney_adj = torch.tensor(A, dtype=torch.float32)
        A = torch.load(heart_path)
        heart_adj = torch.tensor(A, dtype=torch.float32)
        A = torch.load(liver_path)
        liver_adj = torch.tensor(A, dtype=torch.float32)

        return [lung_adj_topk, kidney_adj_topk, heart_adj_topk, liver_adj_topk],[lung_adj, kidney_adj, heart_adj, liver_adj],    lung_kidney_heart_liver_adj

    def compute_cnets(self):
        inter_path = 'data/spatial/graph/inter/'
        cnets = [[None]* self.n_networks for _ in range(self.n_networks)]
        if self.task == 'spatial':
            cts = ['GCBC','NBC_MBC','FDC','epithelial','CD4_T','myeloid']
            #cts = ['GCBC','NBC_MBC','PC','CD4_T']
            for i in range(self.n_networks):
                for j in range(self.n_networks):
                    if i == j:
                        continue
                    cti = cts[i]
                    ctj = cts[j]
                    if i < j:
                        path = os.path.join(inter_path, f'{cti}_{ctj}.pt')
                        A = torch.load(path, weights_only=False).float()
                        cnets[i][j] = A
                    if j <i:
                        path = os.path.join(inter_path, f'{ctj}_{cti}.pt')
                        A = torch.load(path, weights_only=False).float()
                        cnets[i][j] = A.T
        else:
            for i in range(self.n_networks):
                for j in range(self.n_networks):
                    if i == j:
                        continue
                    if i < j:
                        cnets[i][j] = self.cross
                    else:
                        cnets[i][j] = self.cross.T
        return cnets
    def compute_b(self):
        b = [None]*self.n_networks
        for i in range(self.n_networks):
            adj = (self.input[i] > 0).float()
            b[i] = adj.clone()
            b[i] *= self.beta
            b[i] += 1

        return b

    def compute_cnets_masks(self):
        cnets = self.compute_cnets()
        cnets_masks = []
        for i in range(self.n_networks):
            cnets_masks.append([None] * self.n_networks)
            for j in range(self.n_networks):
                if i == j:
                    continue
                tmp = torch.sum(cnets[i][j], dim=1, keepdim=True)  # Ni*1
                tmp[tmp > 0] = 1
                # tmp = np.tile(tmp,(1,n_clusters[i]))
                tmp = tmp.repeat(1, self.n_clusters[i])
                cnets_masks[i][j] = tmp

        return cnets, cnets_masks

    def get_labels(self):

        Gs = [nx.from_numpy_array(x.numpy()) for x in self.xs]
        #Gs = [nx.from_numpy_array(x) for x in self.xs]
        u_is = []
        u_js = []
        u_labels = []
        for i in range(self.n_networks):
            G = Gs[i]
            u_i, u_j, u_label = get_edges(G)
            u_is.append(u_i)
            u_js.append(u_j)
            u_labels.append(u_label)
        return u_is, u_js,u_labels

    def get_data(self):
        cnets, cnets_masks = self.compute_cnets_masks()
        u_is,u_js,u_labels = self.get_labels()
        b = self.compute_b()

        return self.xs,cnets,cnets_masks,u_is,u_js,u_labels,b
