import torch
import logging
import os
import numpy as np
import random
def get_edges(G):
    edges = set()
    u_i = []
    u_j = []
    label = []
    for edge in G.edges():
        if (edge[0],edge[1]) in edges:
            continue
        u_i.append(edge[0])
        u_j.append(edge[1])
        dat = 1 -G.edges[edge[0], edge[1]].get("weight")
        label.append(dat)
        edges.add((edge[0],edge[1]))
    return torch.tensor(u_i), torch.tensor(u_j), torch.tensor(label)

def compare_params(p1, p2):
    if len(p1) != len(p2):
        return False
    for a, b in zip(p1, p2):
        if not torch.equal(a, b):
            return False
    return True

def symmetric(adj):
    for i in range(adj.shape[0]):
        for j in range(i + 1, adj.shape[1]):
            adj[j][i] = adj[i][j]

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def compute_D(adj):
    D = torch.sum(adj, dim=1)
    #D = torch.diag(deg)
    return D

def A_normalized(adj):
    I = np.eye(adj.shape[0])
    a = adj + I
    D = compute_D(a)
    D_inv_sqrt = torch.diag(torch.pow(D, -0.5))
    print(D_inv_sqrt)
    A_normalized = D_inv_sqrt @ a @ D_inv_sqrt

    return A_normalized

def logging_info(path):
    if not os.path.exists(path):
        os.makedirs(path)
    logging.basicConfig(
        filename=os.path.join(path,'loss.log'),
        level=logging.INFO,
        format='%(message)s'
    )

def move_to_cpu(obj):
        if torch.is_tensor(obj):
            return obj.cpu()
        elif isinstance(obj, list):
            return [move_to_cpu(item) for item in obj]
        elif isinstance(obj, dict):
            return {k: move_to_cpu(v) for k, v in obj.items()}
        elif isinstance(obj, tuple):
            return tuple(move_to_cpu(v) for v in obj)
        else:
            return obj
