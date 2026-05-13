from __future__ import annotations

import argparse
import gc
import json
import os
import random
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve
from tqdm import tqdm

from sarem.clean_view import build_clean_graph_by_feature_similarity
from sarem.model import Model
from sarem.reliability import ReliabilityCalibrator
from sarem.substructure import compute_substructure_scores, fuse_original_clean_substructure
from sarem.utils import load_attributed_graph_mat, normalize_adj, row_normalize_features, sample_rwr_subgraphs

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


DEFAULT_CONFIGS = {
    "cora": {"lr": 1e-3, "num_epoch": 100, "subgraph_size": 2},
    "citeseer": {"lr": 1e-3, "num_epoch": 100, "subgraph_size": 2},
    "pubmed": {"lr": 1e-3, "num_epoch": 100, "subgraph_size": 2},
    "ACM": {"lr": 1e-3, "num_epoch": 100, "subgraph_size": 3},
    "Reddit": {"lr": 1e-3, "num_epoch": 100, "subgraph_size": 4},
    "BlogCatalog": {"lr": 3e-3, "num_epoch": 100, "subgraph_size": 5},
    "Flickr": {"lr": 1e-3, "num_epoch": 100, "subgraph_size": 4},
    "books": {"lr": 1e-3, "num_epoch": 400, "subgraph_size": 4},
    "disney": {"lr": 1e-3, "num_epoch": 400, "subgraph_size": 4},
    "enron": {"lr": 1e-3, "num_epoch": 400, "subgraph_size": 4},
}


def parse_csv_arg(text: str) -> List[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["OMP_NUM_THREADS"] = "1"
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def select_device(device_arg: str) -> torch.device:
    if device_arg.startswith("cuda") and torch.cuda.is_available():
        return torch.device(device_arg)
    return torch.device("cpu")


def minmax_np(x, eps: float = 1e-12):
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if len(x) == 0:
        return x
    mn, mx = float(np.min(x)), float(np.max(x))
    if mx - mn < eps:
        return np.zeros_like(x, dtype=np.float64)
    return (x - mn) / (mx - mn + eps)


def finite_float(v):
    try:
        v = float(v)
        return v if np.isfinite(v) else None
    except Exception:
        return None


def float_list(values):
    return [finite_float(x) for x in np.asarray(values).reshape(-1)]


def int_list(values):
    return [int(x) for x in np.asarray(values).reshape(-1)]


def hist_payload(values, bins: int = 60):
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"hist": [], "bin_edges": []}
    hist, bin_edges = np.histogram(arr, bins=bins, density=True)
    return {"hist": float_list(hist), "bin_edges": float_list(bin_edges)}


def compute_metrics(y_true, score) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int).reshape(-1)
    score = np.asarray(score, dtype=np.float64).reshape(-1)
    out = {
        "ROC_AUC": float(roc_auc_score(y_true, score)),
        "PR_AUC": float(average_precision_score(y_true, score)),
    }

    fpr, tpr, _ = roc_curve(y_true, score)
    hit = np.where(tpr >= 0.95)[0]
    out["FPR@95TPR"] = float(fpr[hit[0]]) if len(hit) else float(fpr[-1])

    k = int(y_true.sum())
    if k > 0:
        rank = np.argsort(-score)
        topk = rank[:k]
        tp = int(y_true[topk].sum())
        fp = int(k - tp)
        out["TopK"] = int(k)
        out["Precision@K"] = float(tp / max(k, 1))
        out["Recall@K"] = float(tp / max(k, 1))
        out["FDR@K"] = float(fp / max(k, 1))
        out["FalseAlarms@K"] = int(fp)
    else:
        out.update({"TopK": 0, "Precision@K": None, "Recall@K": None, "FDR@K": None, "FalseAlarms@K": None})
    return out


def false_alarm_analysis(y_true, score, degree, ego_density=None):
    y_true = np.asarray(y_true).astype(int).reshape(-1)
    score = np.asarray(score, dtype=np.float64).reshape(-1)
    degree = np.asarray(degree, dtype=np.float64).reshape(-1)
    k = int(y_true.sum())
    if k <= 0:
        return {}

    rank = np.argsort(-score)
    topk = rank[:k]
    false_nodes = topk[y_true[topk] == 0]
    q30 = float(np.quantile(degree, 0.30))
    q70 = float(np.quantile(degree, 0.70))

    tail_mask = (y_true == 0) & (degree <= q30)
    mid_mask = (y_true == 0) & (degree > q30) & (degree < q70)
    head_mask = (y_true == 0) & (degree >= q70)

    out = {
        "LowDegreeFalseAlarms@K": int(np.sum(degree[false_nodes] <= q30)),
        "FalseAlarmRate@K": float(len(false_nodes) / max(k, 1)),
        "AvgDegreeFalseAlarms@K": float(np.mean(degree[false_nodes])) if len(false_nodes) else 0.0,
        "TailNormalMeanScore": float(np.mean(score[tail_mask])) if np.any(tail_mask) else 0.0,
        "MidNormalMeanScore": float(np.mean(score[mid_mask])) if np.any(mid_mask) else 0.0,
        "HeadNormalMeanScore": float(np.mean(score[head_mask])) if np.any(head_mask) else 0.0,
    }

    if ego_density is not None:
        ego_density = np.asarray(ego_density, dtype=np.float64).reshape(-1)
        den70 = float(np.quantile(ego_density, 0.70))
        high_density_topk = topk[ego_density[topk] >= den70]
        high_density_tp = int(y_true[high_density_topk].sum()) if len(high_density_topk) else 0
        out["HighDensityTopKCount"] = int(len(high_density_topk))
        out["HighDensityAnomTopK"] = int(high_density_tp)
        out["HighDensityAnomPrecisionTopK"] = float(high_density_tp / max(len(high_density_topk), 1))
    return out


def save_curves(y_true, score, out_dir: Path):
    fpr, tpr, roc_thr = roc_curve(y_true, score)
    prec, rec, pr_thr = precision_recall_curve(y_true, score)
    pr_thr_pad = np.concatenate([pr_thr, np.array([np.nan])])

    roc_df = pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": roc_thr})
    pr_df = pd.DataFrame({"precision": prec, "recall": rec, "threshold": pr_thr_pad})
    roc_df.to_csv(out_dir / "roc_curve.csv", index=False)
    pr_df.to_csv(out_dir / "pr_curve.csv", index=False)
    return roc_df, pr_df


def build_batch(idx, adj_t, feat_t, raw_feat_t, subgraphs_1, subgraphs_2, ft_size, subgraph_size, device):
    cur_batch = len(idx)
    ba1, ba2, bf1, bf2, raw_bf1, raw_bf2 = [], [], [], [], [], []

    zero_adj_row = torch.zeros((cur_batch, 1, subgraph_size), device=device)
    zero_adj_col = torch.zeros((cur_batch, subgraph_size + 1, 1), device=device)
    zero_adj_col[:, -1, :] = 1.0
    zero_feat_row = torch.zeros((cur_batch, 1, ft_size), device=device)

    for i in idx:
        s1 = subgraphs_1[i]
        s2 = subgraphs_2[i]
        ba1.append(adj_t[:, s1, :][:, :, s1])
        ba2.append(adj_t[:, s2, :][:, :, s2])
        bf1.append(feat_t[:, s1, :])
        bf2.append(feat_t[:, s2, :])
        raw_bf1.append(raw_feat_t[:, s1, :])
        raw_bf2.append(raw_feat_t[:, s2, :])

    ba1 = torch.cat((torch.cat(ba1), zero_adj_row), dim=1)
    ba1 = torch.cat((ba1, zero_adj_col), dim=2)
    ba2 = torch.cat((torch.cat(ba2), zero_adj_row), dim=1)
    ba2 = torch.cat((ba2, zero_adj_col), dim=2)

    bf1 = torch.cat(bf1)
    bf2 = torch.cat(bf2)
    raw_bf1 = torch.cat(raw_bf1)
    raw_bf2 = torch.cat(raw_bf2)

    bf1 = torch.cat((bf1[:, :-1, :], zero_feat_row, bf1[:, -1:, :]), dim=1)
    bf2 = torch.cat((bf2[:, :-1, :], zero_feat_row, bf2[:, -1:, :]), dim=1)
    raw_bf1 = torch.cat((raw_bf1[:, :-1, :], zero_feat_row, raw_bf1[:, -1:, :]), dim=1)
    raw_bf2 = torch.cat((raw_bf2[:, :-1, :], zero_feat_row, raw_bf2[:, -1:, :]), dim=1)
    return ba1, ba2, bf1, bf2, raw_bf1, raw_bf2


def save_run_plot_json(run_dir: Path, dataset: str, seed: int, metrics: Dict, node_df: pd.DataFrame, roc_df, pr_df, max_scatter_points=3000):
    normal = node_df[node_df["label"] == 0]
    anomaly = node_df[node_df["label"] == 1]

    scatter_df = node_df
    if len(scatter_df) > max_scatter_points:
        scatter_df = scatter_df.sample(max_scatter_points, random_state=1)

    def by_label(col):
        if col not in node_df.columns:
            return {"normal": [], "anomaly": []}
        return {"normal": float_list(normal[col].values), "anomaly": float_list(anomaly[col].values)}

    payload = {
        "metadata": {"dataset": dataset, "seed": int(seed)},
        "metrics": metrics,
        "roc_curve": {
            "fpr": float_list(roc_df["fpr"].values),
            "tpr": float_list(roc_df["tpr"].values),
            "threshold": float_list(roc_df["threshold"].values),
        },
        "pr_curve": {
            "precision": float_list(pr_df["precision"].values),
            "recall": float_list(pr_df["recall"].values),
            "threshold": float_list(pr_df["threshold"].values),
        },
        "distributions": {
            "final_score": {"normal": hist_payload(normal["final_score"]), "anomaly": hist_payload(anomaly["final_score"])},
            "cal_rec_score": {"normal": hist_payload(normal["cal_rec_score"]), "anomaly": hist_payload(anomaly["cal_rec_score"])},
            "reliability": {"normal": hist_payload(normal["reliability"]), "anomaly": hist_payload(anomaly["reliability"])},
            "sub_component": {"normal": hist_payload(normal["sub_component"]), "anomaly": hist_payload(anomaly["sub_component"])},
        },
        "violin_data": {
            "final_score": by_label("final_score"),
            "rec_error": by_label("rec_error"),
            "cal_rec_score": by_label("cal_rec_score"),
            "reliability": by_label("reliability"),
            "sub_score": by_label("sub_score"),
            "sub_reliability": by_label("sub_reliability"),
            "sub_component": by_label("sub_component"),
            "degree": by_label("degree"),
        },
        "scatter": {
            "node_id": int_list(scatter_df["node_id"].values),
            "label": int_list(scatter_df["label"].values),
            "degree": float_list(scatter_df["degree"].values),
            "final_score": float_list(scatter_df["final_score"].values),
            "rec_error": float_list(scatter_df["rec_error"].values),
            "cal_rec_score": float_list(scatter_df["cal_rec_score"].values),
            "reliability": float_list(scatter_df["reliability"].values),
            "ego_density": float_list(scatter_df["ego_density"].values),
            "sub_score": float_list(scatter_df["sub_score"].values),
            "sub_component": float_list(scatter_df["sub_component"].values),
        },
        "output_files": {
            "metrics_json": str(run_dir / "metrics.json"),
            "node_scores_csv": str(run_dir / "node_scores.csv"),
            "roc_curve_csv": str(run_dir / "roc_curve.csv"),
            "pr_curve_csv": str(run_dir / "pr_curve.csv"),
        },
    }
    with open(run_dir / "plot_data.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return payload


def maybe_save_png(run_dir: Path, node_df: pd.DataFrame, roc_df, pr_df, title: str, save_png: bool):
    if not save_png:
        return
    plot_dir = run_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    plt.figure()
    plt.plot(roc_df["fpr"], roc_df["tpr"])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC - {title}")
    plt.tight_layout()
    plt.savefig(plot_dir / "roc_curve.png", dpi=300)
    plt.close()

    plt.figure()
    plt.plot(pr_df["recall"], pr_df["precision"])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"PR - {title}")
    plt.tight_layout()
    plt.savefig(plot_dir / "pr_curve.png", dpi=300)
    plt.close()

    for col, name in [
        ("final_score", "final_score_distribution"),
        ("reliability", "reliability_distribution"),
        ("sub_component", "substructure_component_distribution"),
    ]:
        normal = node_df[node_df["label"] == 0][col]
        anomaly = node_df[node_df["label"] == 1][col]
        plt.figure()
        plt.hist(normal, bins=60, alpha=0.55, density=True, label="Normal")
        plt.hist(anomaly, bins=60, alpha=0.55, density=True, label="Anomaly")
        plt.xlabel(col)
        plt.ylabel("Density")
        plt.title(f"{name} - {title}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_dir / f"{name}.png", dpi=300)
        plt.close()


def run_one(args, dataset: str, seed: int, result_root: Path):
    seed_everything(seed)
    config = DEFAULT_CONFIGS.get(dataset, {"lr": 1e-3, "num_epoch": 100, "subgraph_size": 4})
    lr = args.lr if args.lr is not None else config["lr"]
    num_epoch = args.num_epoch if args.num_epoch is not None else config["num_epoch"]
    subgraph_size = args.subgraph_size if args.subgraph_size is not None else config["subgraph_size"]
    device = select_device(args.device)

    run_dir = result_root / dataset / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[RUN] dataset={dataset} seed={seed} device={device} lr={lr} epochs={num_epoch} subgraph={subgraph_size}")

    adj_sp, features_sp, y_true = load_attributed_graph_mat(dataset, args.data_dir)
    raw_features_np = np.asarray(features_sp.todense(), dtype=np.float32)
    features_np = row_normalize_features(features_sp)

    nb_nodes, ft_size = features_np.shape
    batch_num = nb_nodes // args.batch_size + 1

    adj_norm = normalize_adj(adj_sp)
    adj_dense = np.asarray((adj_norm + sp.eye(adj_norm.shape[0])).todense(), dtype=np.float32)
    features_t = torch.FloatTensor(features_np[np.newaxis]).to(device)
    raw_features_t = torch.FloatTensor(raw_features_np[np.newaxis]).to(device)
    adj_t = torch.FloatTensor(adj_dense[np.newaxis]).to(device)

    calibrator = ReliabilityCalibrator(
        num_nodes=nb_nodes,
        device=str(device),
        momentum=args.rel_momentum,
        floor=args.rel_floor,
        w_deg=args.w_deg,
        w_nei=args.w_nei,
        w_dyn=args.w_dyn,
    ).fit_static(adj_sp, features_np)

    print(f"[1/4] Computing original ego-substructure evidence: {dataset}/seed{seed}", flush=True)
    raw_sub_tuple = compute_substructure_scores(
        adj_sp,
        features_np,
        max_neighbors=args.max_sub_neighbors,
        show_progress=True,
        desc=f"SubRaw {dataset}/s{seed}",
    )
    print(f"[2/4] Building clean graph: {dataset}/seed{seed}", flush=True)
    clean_adj_sp, clean_info = build_clean_graph_by_feature_similarity(adj_sp, features_np, drop_ratio=args.clean_drop_ratio)
    print(f"[3/4] Computing clean-view ego-substructure evidence: {dataset}/seed{seed}", flush=True)
    clean_sub_tuple = compute_substructure_scores(
        clean_adj_sp,
        features_np,
        max_neighbors=args.max_sub_neighbors,
        show_progress=True,
        desc=f"SubClean {dataset}/s{seed}",
    )
    print(f"[4/4] Starting model training/testing: {dataset}/seed{seed}", flush=True)
    sub_score, sub_rel, ego_density, sub_similarity, ego_size = fuse_original_clean_substructure(
        raw_sub_tuple, clean_sub_tuple, clean_weight=args.clean_sub_weight
    )
    sub_score_raw, sub_rel_raw, ego_density_raw, sub_similarity_raw, ego_size_raw = raw_sub_tuple
    sub_score_clean, sub_rel_clean, ego_density_clean, sub_similarity_clean, ego_size_clean = clean_sub_tuple

    with open(run_dir / "clean_view_info.json", "w", encoding="utf-8") as f:
        json.dump(clean_info, f, indent=2, ensure_ascii=False)

    model = Model(ft_size, args.embedding_dim, "prelu", args.negsamp_ratio, args.readout).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=args.weight_decay)
    b_xent = nn.BCEWithLogitsLoss(
        reduction="none",
        pos_weight=torch.tensor([args.negsamp_ratio], dtype=torch.float32, device=device),
    )

    use_reliability_train = bool(args.use_reliability_train)
    best, best_epoch, cnt_wait = 1e18, 0, 0
    ckpt_path = run_dir / "best_model.pkl"
    best_cal_state = calibrator.snapshot()
    train_log = []

    for epoch in tqdm(range(num_epoch), desc=f"Train {dataset}/s{seed}", dynamic_ncols=True, file=sys.stdout, mininterval=0.5):
        model.train()
        all_idx = list(range(nb_nodes))
        random.shuffle(all_idx)
        subgraphs_1 = sample_rwr_subgraphs(adj_sp, subgraph_size, restart_prob=args.restart_prob)
        subgraphs_2 = sample_rwr_subgraphs(adj_sp, subgraph_size, restart_prob=args.restart_prob)

        total_loss, total_l1, total_l2, total_seen = 0.0, 0.0, 0.0, 0
        for batch_idx in range(batch_num):
            idx = all_idx[batch_idx * args.batch_size: min((batch_idx + 1) * args.batch_size, nb_nodes)]
            if not idx:
                continue
            cur_batch = len(idx)
            idx_t = torch.tensor(idx, dtype=torch.long, device=device)
            labels = torch.cat((torch.ones(cur_batch), torch.zeros(cur_batch * args.negsamp_ratio))).view(-1, 1).to(device)

            ba1, ba2, bf1, bf2, raw_bf1, raw_bf2 = build_batch(
                idx, adj_t, features_t, raw_features_t, subgraphs_1, subgraphs_2, ft_size, subgraph_size, device
            )
            logits, f_1, f_2 = model(bf1, bf2, raw_bf1, raw_bf2, ba1, ba2)
            loss1 = torch.mean(b_xent(logits, labels))
            rec_err1 = (f_1[:, -2, :] - raw_bf1[:, -1, :]).pow(2).mean(dim=1)
            rec_err2 = (f_2[:, -2, :] - raw_bf2[:, -1, :]).pow(2).mean(dim=1)
            rec_error = 0.5 * (rec_err1 + rec_err2)

            if use_reliability_train and epoch >= args.rel_warmup:
                weight = calibrator.pseudo_normal_weight(idx_t, temperature=args.rel_temp)
                loss2 = (weight * rec_error).sum() / (weight.sum() + 1e-12)
            else:
                loss2 = rec_error.mean()

            loss = args.alpha * loss1 + args.beta * loss2
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            calibrator.update_dynamic(idx_t, rec_error.detach())

            total_loss += float(loss.detach().cpu()) * cur_batch
            total_l1 += float(loss1.detach().cpu()) * cur_batch
            total_l2 += float(loss2.detach().cpu()) * cur_batch
            total_seen += cur_batch

        mean_loss = total_loss / max(total_seen, 1)
        row = {
            "epoch": int(epoch),
            "loss": mean_loss,
            "contrastive_loss": total_l1 / max(total_seen, 1),
            "reconstruction_loss": total_l2 / max(total_seen, 1),
            "best_epoch": int(best_epoch),
        }
        train_log.append(row)

        if mean_loss < best:
            best, best_epoch, cnt_wait = mean_loss, epoch, 0
            torch.save(model.state_dict(), ckpt_path)
            best_cal_state = calibrator.snapshot()
        else:
            cnt_wait += 1
        if cnt_wait >= args.patience:
            print(f"[Early Stop] epoch={epoch}, best_epoch={best_epoch}")
            break

    pd.DataFrame(train_log).to_csv(run_dir / "train_log.csv", index=False)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    calibrator.load_snapshot(best_cal_state)
    model.eval()

    all_score = np.zeros((args.auc_test_rounds, nb_nodes), dtype=np.float64)
    all_con = np.zeros_like(all_score)
    all_rec = np.zeros_like(all_score)
    all_cal_rec = np.zeros_like(all_score)
    all_rel = np.zeros_like(all_score)
    all_sub = np.zeros_like(all_score)

    with torch.no_grad():
        for round_id in tqdm(range(args.auc_test_rounds), desc=f"Test {dataset}/s{seed}", leave=True, dynamic_ncols=True, file=sys.stdout, mininterval=0.5):
            all_idx = list(range(nb_nodes))
            random.shuffle(all_idx)
            subgraphs_1 = sample_rwr_subgraphs(adj_sp, subgraph_size, restart_prob=args.restart_prob)
            subgraphs_2 = sample_rwr_subgraphs(adj_sp, subgraph_size, restart_prob=args.restart_prob)

            for batch_idx in range(batch_num):
                idx = all_idx[batch_idx * args.batch_size: min((batch_idx + 1) * args.batch_size, nb_nodes)]
                if not idx:
                    continue
                cur_batch = len(idx)
                idx_t = torch.tensor(idx, dtype=torch.long, device=device)

                ba1, ba2, bf1, bf2, raw_bf1, raw_bf2 = build_batch(
                    idx, adj_t, features_t, raw_features_t, subgraphs_1, subgraphs_2, ft_size, subgraph_size, device
                )
                logits, dist = model.inference(bf1, bf2, raw_bf1, raw_bf2, ba1, ba2)
                logits = torch.sigmoid(torch.squeeze(logits))

                if args.negsamp_ratio == 1:
                    con_score = -(logits[:cur_batch] - logits[cur_batch:]).detach().cpu().numpy()
                else:
                    pos = logits[:cur_batch]
                    neg = logits[cur_batch:].view(-1, cur_batch).mean(dim=0)
                    con_score = -(pos - neg).detach().cpu().numpy()

                rec_score = dist.detach().cpu().numpy()
                reliability = calibrator.reliability(idx_t).detach().cpu().numpy()
                cal_rec = reliability * rec_score
                sub_component = sub_score[idx] * sub_rel[idx]

                con_norm = minmax_np(con_score)
                cal_rec_norm = minmax_np(cal_rec)
                sub_norm = minmax_np(sub_component)
                final_score = args.alpha * con_norm + args.beta * cal_rec_norm + args.gamma * sub_norm

                all_score[round_id, idx] = final_score
                all_con[round_id, idx] = con_score
                all_rec[round_id, idx] = rec_score
                all_cal_rec[round_id, idx] = cal_rec
                all_rel[round_id, idx] = reliability
                all_sub[round_id, idx] = sub_component

    final_score = all_score.mean(axis=0)
    final_con = all_con.mean(axis=0)
    final_rec = all_rec.mean(axis=0)
    final_cal_rec = all_cal_rec.mean(axis=0)
    final_rel = all_rel.mean(axis=0)
    final_sub_component = all_sub.mean(axis=0)

    rel_arrays = calibrator.numpy_all()
    metrics = compute_metrics(y_true, final_score)
    metrics.update(false_alarm_analysis(y_true, final_score, rel_arrays["degree"], ego_density=ego_density))
    metrics.update({
        "dataset": dataset,
        "seed": int(seed),
        "lr": float(lr),
        "num_epoch": int(num_epoch),
        "best_epoch": int(best_epoch),
        "subgraph_size": int(subgraph_size),
        "batch_size": int(args.batch_size),
        "auc_test_rounds": int(args.auc_test_rounds),
        "alpha": float(args.alpha),
        "beta": float(args.beta),
        "gamma": float(args.gamma),
        "clean_drop_ratio": float(args.clean_drop_ratio),
        "clean_sub_weight": float(args.clean_sub_weight),
        "use_reliability_train": bool(use_reliability_train),
    })

    node_df = pd.DataFrame({
        "node_id": np.arange(nb_nodes),
        "label": y_true,
        "degree": rel_arrays["degree"],
        "q_deg": rel_arrays["q_deg"],
        "q_nei": rel_arrays["q_nei"],
        "q_dyn": rel_arrays["q_dyn"],
        "reliability": final_rel,
        "con_score": final_con,
        "rec_error": final_rec,
        "cal_rec_score": final_cal_rec,
        "sub_score_raw": sub_score_raw,
        "sub_rel_raw": sub_rel_raw,
        "sub_score_clean": sub_score_clean,
        "sub_rel_clean": sub_rel_clean,
        "sub_score": sub_score,
        "sub_reliability": sub_rel,
        "sub_component": final_sub_component,
        "ego_density": ego_density,
        "sub_similarity": sub_similarity,
        "ego_size": ego_size,
        "ego_density_raw": ego_density_raw,
        "ego_density_clean": ego_density_clean,
        "sub_similarity_raw": sub_similarity_raw,
        "sub_similarity_clean": sub_similarity_clean,
        "final_score": final_score,
    })
    node_df.to_csv(run_dir / "node_scores.csv", index=False)

    with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    roc_df, pr_df = save_curves(y_true, final_score, run_dir)
    run_plot = save_run_plot_json(run_dir, dataset, seed, metrics, node_df, roc_df, pr_df)
    maybe_save_png(run_dir, node_df, roc_df, pr_df, f"{dataset}-seed{seed}", args.save_png)

    print(
        f"[DONE] {dataset:12s} seed={seed} "
        f"AUC={metrics['ROC_AUC']:.4f} AP={metrics['PR_AUC']:.4f} "
        f"FPR95={metrics['FPR@95TPR']:.4f} FDR@K={metrics['FDR@K']:.4f}"
    )

    del model, optimiser, features_t, raw_features_t, adj_t
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    return metrics, run_plot


def aggregate_summary(result_root: Path):
    rows = []
    for path in result_root.glob("*/seed_*/metrics.json"):
        with open(path, "r", encoding="utf-8") as f:
            rows.append(json.load(f))
    if not rows:
        return None, None
    df = pd.DataFrame(rows)
    df.to_csv(result_root / "summary_runs.csv", index=False)

    metric_cols = [
        "ROC_AUC", "PR_AUC", "FPR@95TPR", "FDR@K", "FalseAlarmRate@K",
        "LowDegreeFalseAlarms@K", "HighDensityAnomPrecisionTopK",
    ]
    metric_cols = [c for c in metric_cols if c in df.columns]
    summary = df.groupby(["dataset"])[metric_cols].agg(["mean", "std"]).reset_index()
    summary.columns = ["_".join([str(x) for x in col if x != ""]).rstrip("_") if isinstance(col, tuple) else str(col) for col in summary.columns]
    summary.to_csv(result_root / "summary_mean_std.csv", index=False)
    return df, summary


def save_dataset_plot_json(result_root: Path, dataset: str, run_payloads: List[Dict], args, timestamp: str):
    plot_root = result_root / "plot_json"
    plot_root.mkdir(parents=True, exist_ok=True)
    json_path = plot_root / f"{dataset}_{timestamp}.json"


    rows = [p["metrics"] for p in run_payloads]
    df = pd.DataFrame(rows) if rows else pd.DataFrame()
    summary_records = []
    if not df.empty:
        metrics = [c for c in ["ROC_AUC", "PR_AUC", "FPR@95TPR", "FDR@K", "FalseAlarmRate@K", "LowDegreeFalseAlarms@K"] if c in df.columns]
        summary = df.groupby("dataset")[metrics].agg(["mean", "std"]).reset_index()
        summary.columns = ["_".join([str(x) for x in col if x != ""]).rstrip("_") if isinstance(col, tuple) else str(col) for col in summary.columns]
        summary_records = summary.to_dict(orient="records")

    payload = {
        "metadata": {
            "dataset": dataset,
            "time": timestamp,
            "file_name_rule": "<dataset>_<YYYYMMDD_HHMMSS>.json",
            "description": "Dataset-level plotting JSON containing all seeds for this dataset.",
        },
        "run_config": vars(args),
        "summary": summary_records,
        "runs": run_payloads,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return json_path


def main():
    parser = argparse.ArgumentParser(description="SAREM graph anomaly detection runner")
    parser.add_argument("--data_dir", type=str, default="./dataset")
    parser.add_argument("--result_root", type=str, default="./results")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--datasets", type=str, default="cora,citeseer,pubmed,ACM,BlogCatalog,Flickr")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--seed_start", type=int, default=1)

    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--num_epoch", type=int, default=None)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--embedding_dim", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--subgraph_size", type=int, default=None)
    parser.add_argument("--readout", type=str, default="avg")
    parser.add_argument("--auc_test_rounds", type=int, default=64)
    parser.add_argument("--negsamp_ratio", type=int, default=1)
    parser.add_argument("--restart_prob", type=float, default=0.9)

    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.6)
    parser.add_argument("--gamma", type=float, default=0.3)

    parser.add_argument("--rel_warmup", type=int, default=20)
    parser.add_argument("--rel_momentum", type=float, default=0.9)
    parser.add_argument("--rel_floor", type=float, default=0.2)
    parser.add_argument("--rel_temp", type=float, default=0.5)
    parser.add_argument("--w_deg", type=float, default=0.3)
    parser.add_argument("--w_nei", type=float, default=0.4)
    parser.add_argument("--w_dyn", type=float, default=0.3)
    parser.add_argument("--use_reliability_train_for_sarem", action="store_true")

    parser.add_argument("--clean_drop_ratio", type=float, default=0.1)
    parser.add_argument("--clean_sub_weight", type=float, default=0.5)
    parser.add_argument("--max_sub_neighbors", type=int, default=None)

    parser.add_argument("--save_png", action="store_true", help="Also save PNG plots. JSON plotting data is always saved.")
    args = parser.parse_args()

    datasets = parse_csv_arg(args.datasets)
    seeds = [args.seed_start + i for i in range(args.runs)]

    result_root = Path(args.result_root)
    result_root.mkdir(parents=True, exist_ok=True)
    with open(result_root / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    all_rows = []
    dataset_json_paths = []

    for dataset in datasets:
        dataset_payloads = []
        dataset_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        for seed in seeds:
            try:
                metrics, plot_payload = run_one(args, dataset, seed, result_root)
                all_rows.append(metrics)
                dataset_payloads.append(plot_payload)
                pd.DataFrame(all_rows).to_csv(result_root / "summary_runs_live.csv", index=False)
                aggregate_summary(result_root)
            except FileNotFoundError as e:
                print(f"[SKIP] dataset={dataset}: {e}")
                break
            except KeyboardInterrupt:
                print("\n[INTERRUPTED] User stopped the run.")
                raise
            except Exception as e:
                print(f"[ERROR] dataset={dataset}, seed={seed}: {e}")
                traceback.print_exc()

        if dataset_payloads:
            json_path = save_dataset_plot_json(result_root, dataset, dataset_payloads, args, dataset_timestamp)
            dataset_json_paths.append(str(json_path))
            print(f"[PLOT_JSON] {json_path}")

    df, summary = aggregate_summary(result_root)
    with open(result_root / "dataset_plot_json_files.json", "w", encoding="utf-8") as f:
        json.dump({"files": dataset_json_paths}, f, indent=2, ensure_ascii=False)

    print("\nAll experiments finished.")
    if summary is not None:
        print(summary)
    print(f"Dataset-level plotting JSON files are saved in: {result_root / 'plot_json'}")


if __name__ == "__main__":
    main()
