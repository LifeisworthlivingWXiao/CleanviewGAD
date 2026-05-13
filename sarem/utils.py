from __future__ import annotations

import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import scipy.io as sio
import scipy.sparse as sp


def load_attributed_graph_mat(dataset: str, data_dir: str = "./dataset"):
    path = Path(data_dir) / f"{dataset}.mat"
    if not path.exists():
        raise FileNotFoundError(f"Cannot find dataset file: {path}")

    data = sio.loadmat(str(path))
    if "Attributes" in data:
        attr = data["Attributes"]
    elif "X" in data:
        attr = data["X"]
    else:
        raise KeyError(f"{path} has no feature key. Expected 'Attributes' or 'X'.")

    if "Network" in data:
        network = data["Network"]
    elif "A" in data:
        network = data["A"]
    else:
        raise KeyError(f"{path} has no graph key. Expected 'Network' or 'A'.")

    if "Label" in data:
        label = data["Label"]
    elif "gnd" in data:
        label = data["gnd"]
    elif "label" in data:
        label = data["label"]
    else:
        raise KeyError(f"{path} has no anomaly label key. Expected 'Label', 'gnd', or 'label'.")

    adj = sp.csr_matrix(network).astype(np.float32)
    feat = attr if sp.issparse(attr) else sp.csr_matrix(attr)
    feat = sp.lil_matrix(feat).astype(np.float32)

    y = np.squeeze(np.asarray(label)).astype(int)
    uniq = np.unique(y)
    if uniq.size > 0 and uniq.min() > 0:
        y = y - uniq.min()
    y = (y > 0).astype(int)
    return adj, feat, y


def row_normalize_features(features):
    if sp.issparse(features):
        rowsum = np.asarray(features.sum(1)).reshape(-1)
        inv = np.power(rowsum, -1, where=rowsum != 0)
        inv[~np.isfinite(inv)] = 0.0
        mat = sp.diags(inv).dot(features)
        return np.asarray(mat.todense(), dtype=np.float32)

    x = np.asarray(features, dtype=np.float32)
    rowsum = x.sum(axis=1, keepdims=True)
    rowsum[rowsum == 0] = 1.0
    return x / rowsum


def normalize_adj(adj):
    adj = sp.coo_matrix(adj).astype(np.float32)
    rowsum = np.asarray(adj.sum(1)).reshape(-1)
    d_inv_sqrt = np.power(rowsum, -0.5, where=rowsum != 0)
    d_inv_sqrt[~np.isfinite(d_inv_sqrt)] = 0.0
    d_mat = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat).transpose().dot(d_mat).tocoo()


def sample_rwr_subgraphs(adj_sp, subgraph_size: int, restart_prob: float = 0.9) -> List[List[int]]:
    adj = sp.csr_matrix(adj_sp)
    n = adj.shape[0]
    k = max(int(subgraph_size) - 1, 0)
    if k == 0:
        return [[i] for i in range(n)]

    indptr, indices = adj.indptr, adj.indices
    all_nodes = list(range(n))
    subgraphs: List[List[int]] = []

    for i in range(n):
        selected = []
        selected_set = set()
        cur = i
        max_steps = max(20, subgraph_size * 20)

        for _ in range(max_steps):
            if len(selected) >= k:
                break
            neigh = indices[indptr[cur]:indptr[cur + 1]]
            if len(neigh) == 0 or random.random() < restart_prob:
                cur = i
            else:
                cur = int(random.choice(neigh))
                if cur != i and cur not in selected_set:
                    selected.append(cur)
                    selected_set.add(cur)

        if len(selected) < k:
            direct = list(indices[indptr[i]:indptr[i + 1]])
            random.shuffle(direct)
            for v in direct:
                v = int(v)
                if len(selected) >= k:
                    break
                if v != i and v not in selected_set:
                    selected.append(v)
                    selected_set.add(v)

        while len(selected) < k:
            if selected:
                selected.append(random.choice(selected))
            elif n > 1:
                v = random.choice(all_nodes)
                selected.append(i if v == i else v)
            else:
                selected.append(i)

        subgraphs.append(selected[:k] + [i])

    return subgraphs
