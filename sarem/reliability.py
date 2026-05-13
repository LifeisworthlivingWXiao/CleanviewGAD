import copy
import numpy as np
import scipy.sparse as sp
import torch


class ReliabilityCalibrator:

    def __init__(
        self,
        num_nodes,
        device="cuda:0",
        momentum=0.9,
        floor=0.2,
        w_deg=0.3,
        w_nei=0.4,
        w_dyn=0.3,
        eps=1e-12,
    ):
        self.num_nodes = int(num_nodes)
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.momentum = float(momentum)
        self.floor = float(floor)
        self.w_deg = float(w_deg)
        self.w_nei = float(w_nei)
        self.w_dyn = float(w_dyn)
        self.eps = float(eps)

        self.q_deg = torch.ones(self.num_nodes, device=self.device)
        self.q_nei = torch.ones(self.num_nodes, device=self.device)
        self.ema_err = torch.zeros(self.num_nodes, device=self.device)
        self.ema_err2 = torch.zeros(self.num_nodes, device=self.device)
        self.seen = torch.zeros(self.num_nodes, device=self.device)

        self.degree_np = np.ones(self.num_nodes, dtype=np.float32)

    def fit_static(self, adj_sp, feat_np):
        if not sp.issparse(adj_sp):
            adj_sp = sp.csr_matrix(adj_sp)
        adj_sp = adj_sp.tocsr()
        feat_np = np.asarray(feat_np, dtype=np.float32)

        deg = np.asarray(adj_sp.sum(axis=1)).reshape(-1).astype(np.float32)
        self.degree_np = deg.copy()


        q_deg = np.log1p(deg)
        q_deg = q_deg / (q_deg.max() + self.eps)
        q_deg = np.clip(q_deg, 0.0, 1.0)


        adj_loop = (adj_sp + sp.eye(adj_sp.shape[0], format="csr")).tocsr()
        deg_loop = np.asarray(adj_loop.sum(axis=1)).reshape(-1).astype(np.float32)
        deg_loop[deg_loop <= 0] = 1.0
        neigh_feat = adj_loop.dot(feat_np) / deg_loop[:, None]

        feat_norm = np.linalg.norm(feat_np, axis=1)
        neigh_norm = np.linalg.norm(neigh_feat, axis=1)
        denom = feat_norm * neigh_norm + self.eps
        sim = np.sum(feat_np * neigh_feat, axis=1) / denom
        q_nei = (sim + 1.0) / 2.0
        q_nei = np.nan_to_num(q_nei, nan=0.5, posinf=1.0, neginf=0.0)
        q_nei = np.clip(q_nei, 0.0, 1.0)

        self.q_deg = torch.tensor(q_deg, dtype=torch.float32, device=self.device)
        self.q_nei = torch.tensor(q_nei, dtype=torch.float32, device=self.device)
        return self

    def _ids(self, node_ids):
        if isinstance(node_ids, torch.Tensor):
            return node_ids.long().to(self.device)
        return torch.tensor(node_ids, dtype=torch.long, device=self.device)

    @torch.no_grad()
    def update_dynamic(self, node_ids, rec_error):
        node_ids = self._ids(node_ids)
        rec_error = rec_error.detach().float().to(self.device).view(-1)

        old_m = self.ema_err[node_ids]
        old_m2 = self.ema_err2[node_ids]
        old_seen = self.seen[node_ids]

        new_m = self.momentum * old_m + (1.0 - self.momentum) * rec_error
        new_m2 = self.momentum * old_m2 + (1.0 - self.momentum) * rec_error.pow(2)

        first_seen = old_seen <= 0
        if first_seen.any():
            new_m[first_seen] = rec_error[first_seen]
            new_m2[first_seen] = rec_error[first_seen].pow(2)

        self.ema_err[node_ids] = new_m
        self.ema_err2[node_ids] = new_m2
        self.seen[node_ids] = 1.0

    def dynamic_reliability(self, node_ids):
        node_ids = self._ids(node_ids)
        m = self.ema_err[node_ids]
        m2 = self.ema_err2[node_ids]
        var = (m2 - m.pow(2)).clamp(min=0.0)
        std = torch.sqrt(var + self.eps)


        q_dyn = torch.exp(-std / (m.abs() + self.eps))
        q_dyn = torch.where(self.seen[node_ids] > 0, q_dyn, torch.ones_like(q_dyn))
        return q_dyn.clamp(0.0, 1.0)

    def reliability(self, node_ids, use_deg=True, use_nei=True, use_dyn=True):
        node_ids = self._ids(node_ids)

        active_weights = []
        parts = []

        if use_deg and self.w_deg > 0:
            parts.append(self.w_deg * self.q_deg[node_ids])
            active_weights.append(self.w_deg)
        if use_nei and self.w_nei > 0:
            parts.append(self.w_nei * self.q_nei[node_ids])
            active_weights.append(self.w_nei)
        if use_dyn and self.w_dyn > 0:
            parts.append(self.w_dyn * self.dynamic_reliability(node_ids))
            active_weights.append(self.w_dyn)

        if len(parts) == 0:
            base = torch.ones(len(node_ids), device=self.device)
        else:
            denom = sum(active_weights) + self.eps
            base = sum(parts) / denom

        r = self.floor + (1.0 - self.floor) * base
        return r.clamp(self.floor, 1.0)

    def pseudo_normal_weight(
        self,
        node_ids,
        temperature=0.5,
        use_deg=True,
        use_nei=True,
        use_dyn=True,
    ):
        node_ids = self._ids(node_ids)
        r = self.reliability(node_ids, use_deg=use_deg, use_nei=use_nei, use_dyn=use_dyn)

        hist = self.ema_err[node_ids].detach()
        if torch.max(hist) > torch.min(hist):
            hist_norm = (hist - hist.min()) / (hist.max() - hist.min() + self.eps)
        else:
            hist_norm = torch.zeros_like(hist)

        pseudo_normal = torch.exp(-hist_norm / max(float(temperature), self.eps))
        weight = r * pseudo_normal
        return weight.detach().clamp(min=self.floor, max=1.0)

    def numpy_all(self, use_deg=True, use_nei=True, use_dyn=True):
        ids = torch.arange(self.num_nodes, device=self.device)
        r = self.reliability(ids, use_deg=use_deg, use_nei=use_nei, use_dyn=use_dyn)
        q_dyn = self.dynamic_reliability(ids)
        return {
            "degree": self.degree_np.copy(),
            "q_deg": self.q_deg.detach().cpu().numpy(),
            "q_nei": self.q_nei.detach().cpu().numpy(),
            "q_dyn": q_dyn.detach().cpu().numpy(),
            "reliability": r.detach().cpu().numpy(),
            "ema_err": self.ema_err.detach().cpu().numpy(),
        }

    def snapshot(self):
        return {
            "q_deg": self.q_deg.detach().clone(),
            "q_nei": self.q_nei.detach().clone(),
            "ema_err": self.ema_err.detach().clone(),
            "ema_err2": self.ema_err2.detach().clone(),
            "seen": self.seen.detach().clone(),
            "degree_np": self.degree_np.copy(),
        }

    def load_snapshot(self, state):
        self.q_deg = state["q_deg"].detach().clone().to(self.device)
        self.q_nei = state["q_nei"].detach().clone().to(self.device)
        self.ema_err = state["ema_err"].detach().clone().to(self.device)
        self.ema_err2 = state["ema_err2"].detach().clone().to(self.device)
        self.seen = state["seen"].detach().clone().to(self.device)
        self.degree_np = copy.deepcopy(state["degree_np"])
        return self
