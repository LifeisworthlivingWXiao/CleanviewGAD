import numpy as np
import scipy.sparse as sp


def _row_normalize_np(x, eps=1e-12):
    x = np.asarray(x, dtype=np.float32)
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / (norm + eps)


def build_clean_graph_by_feature_similarity(adj_sp, feat_np, drop_ratio=0.1, keep_self_loop=False):
    if not sp.issparse(adj_sp):
        adj_sp = sp.csr_matrix(adj_sp)
    adj = adj_sp.tocoo()
    feat_np = np.asarray(feat_np, dtype=np.float32)

    rows = adj.row
    cols = adj.col
    data = adj.data

    if len(rows) == 0:
        return adj_sp.tocsr(), {
            "num_edges_original_directed": 0,
            "num_edges_clean_directed": 0,
            "drop_ratio": float(drop_ratio),
            "threshold": None,
        }

    if not keep_self_loop:
        mask_non_self = rows != cols
        rows = rows[mask_non_self]
        cols = cols[mask_non_self]
        data = data[mask_non_self]

    if len(rows) == 0:
        clean_adj = sp.csr_matrix(adj_sp.shape, dtype=np.float32)
        return clean_adj, {
            "num_edges_original_directed": int(adj.nnz),
            "num_edges_clean_directed": 0,
            "drop_ratio": float(drop_ratio),
            "threshold": None,
        }

    feat_unit = _row_normalize_np(feat_np)
    sims = np.sum(feat_unit[rows] * feat_unit[cols], axis=1)
    sims = np.nan_to_num(sims, nan=0.0, posinf=1.0, neginf=-1.0)

    drop_ratio = float(np.clip(drop_ratio, 0.0, 0.95))
    threshold = np.quantile(sims, drop_ratio)
    keep = sims >= threshold

    clean_adj = sp.coo_matrix(
        (data[keep], (rows[keep], cols[keep])),
        shape=adj_sp.shape,
        dtype=np.float32,
    ).tocsr()


    clean_adj = clean_adj.maximum(clean_adj.T).tocsr()
    clean_adj.eliminate_zeros()

    edge_info = {
        "num_edges_original_directed": int(adj.nnz),
        "num_edges_clean_directed": int(clean_adj.nnz),
        "drop_ratio": float(drop_ratio),
        "threshold": float(threshold),
        "mean_edge_similarity": float(np.mean(sims)),
        "min_edge_similarity": float(np.min(sims)),
        "max_edge_similarity": float(np.max(sims)),
    }
    return clean_adj, edge_info


def build_clean_graph_by_reliability(adj_sp, feat_np, node_reliability, drop_ratio=0.1):
    if not sp.issparse(adj_sp):
        adj_sp = sp.csr_matrix(adj_sp)
    adj = adj_sp.tocoo()
    feat_np = np.asarray(feat_np, dtype=np.float32)
    node_reliability = np.asarray(node_reliability, dtype=np.float32).reshape(-1)

    rows = adj.row
    cols = adj.col
    data = adj.data

    mask_non_self = rows != cols
    rows = rows[mask_non_self]
    cols = cols[mask_non_self]
    data = data[mask_non_self]

    if len(rows) == 0:
        return sp.csr_matrix(adj_sp.shape, dtype=np.float32), {}

    feat_unit = _row_normalize_np(feat_np)
    sims = np.sum(feat_unit[rows] * feat_unit[cols], axis=1)
    sims = np.nan_to_num(sims, nan=0.0, posinf=1.0, neginf=-1.0)

    rel_edge = np.sqrt(
        np.clip(node_reliability[rows], 0.0, 1.0)
        * np.clip(node_reliability[cols], 0.0, 1.0)
    )

    edge_score = sims * rel_edge
    threshold = np.quantile(edge_score, np.clip(drop_ratio, 0.0, 0.95))
    keep = edge_score >= threshold

    clean_adj = sp.coo_matrix(
        (data[keep], (rows[keep], cols[keep])),
        shape=adj_sp.shape,
        dtype=np.float32,
    ).tocsr()
    clean_adj = clean_adj.maximum(clean_adj.T).tocsr()
    clean_adj.eliminate_zeros()

    return clean_adj, {
        "threshold": float(threshold),
        "mean_edge_score": float(np.mean(edge_score)),
        "num_edges_original_directed": int(adj.nnz),
        "num_edges_clean_directed": int(clean_adj.nnz),
    }
