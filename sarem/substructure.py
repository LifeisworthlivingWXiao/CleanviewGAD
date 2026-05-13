import sys

import numpy as np
import scipy.sparse as sp
from tqdm import tqdm


def _row_normalize_np(x, eps=1e-12):
    x = np.asarray(x, dtype=np.float32)
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / (norm + eps)


def compute_substructure_scores(adj_sp, feat_np, max_neighbors=None, show_progress=True, desc="Substructure"):
    if not sp.issparse(adj_sp):
        adj_sp = sp.csr_matrix(adj_sp)
    adj = adj_sp.tocsr()
    adj = adj.maximum(adj.T).tocsr()
    adj.setdiag(0)
    adj.eliminate_zeros()

    feat_np = np.asarray(feat_np, dtype=np.float32)
    feat_unit = _row_normalize_np(feat_np)

    n = adj.shape[0]
    sub_score = np.zeros(n, dtype=np.float32)
    sub_reliability = np.zeros(n, dtype=np.float32)
    ego_density = np.zeros(n, dtype=np.float32)
    sub_similarity = np.ones(n, dtype=np.float32)
    ego_size = np.ones(n, dtype=np.float32)

    rng = np.random.default_rng(1)

    iterator = tqdm(
        range(n),
        desc=desc,
        dynamic_ncols=True,
        mininterval=0.5,
        file=sys.stdout,
        disable=not show_progress,
    )

    for i in iterator:
        neigh = adj[i].indices
        if max_neighbors is not None and len(neigh) > max_neighbors:
            neigh = rng.choice(neigh, size=max_neighbors, replace=False)

        if len(neigh) == 0:
            ego_density[i] = 0.0
            sub_similarity[i] = 1.0
            sub_score[i] = 0.0
            sub_reliability[i] = 0.0
            ego_size[i] = 1.0
            continue

        nodes = np.unique(np.concatenate([neigh, np.array([i], dtype=neigh.dtype)]))
        k = len(nodes)
        ego_size[i] = float(k)

        if k <= 1:
            continue

        sub_adj = adj[nodes][:, nodes]

        undirected_edges = sub_adj.nnz / 2.0
        density = 2.0 * undirected_edges / (k * (k - 1) + 1e-12)
        density = float(np.clip(density, 0.0, 1.0))
        ego_density[i] = density

        center = feat_unit[i]
        others = feat_unit[nodes]
        sims = others @ center


        self_mask = nodes == i
        if np.any(self_mask) and k > 1:
            sim_mean = (np.sum(sims) - sims[self_mask][0]) / max(k - 1, 1)
        else:
            sim_mean = np.mean(sims)

        sim_mean = float(np.clip(sim_mean, -1.0, 1.0))
        sub_similarity[i] = sim_mean


        inconsistency = (1.0 - sim_mean) / 2.0
        sub_score[i] = density * inconsistency


        size_factor = min(k / 10.0, 1.0)
        sub_reliability[i] = density * size_factor

    return sub_score, sub_reliability, ego_density, sub_similarity, ego_size


def fuse_original_clean_substructure(
    raw_tuple,
    clean_tuple,
    clean_weight=0.5,
):
    clean_weight = float(np.clip(clean_weight, 0.0, 1.0))
    raw_weight = 1.0 - clean_weight

    raw_score, raw_rel, raw_den, raw_sim, raw_size = raw_tuple
    clean_score, clean_rel, clean_den, clean_sim, clean_size = clean_tuple

    score = raw_weight * raw_score + clean_weight * clean_score
    rel = raw_weight * raw_rel + clean_weight * clean_rel
    density = raw_weight * raw_den + clean_weight * clean_den
    similarity = raw_weight * raw_sim + clean_weight * clean_sim
    size = raw_weight * raw_size + clean_weight * clean_size

    return score, rel, density, similarity, size
