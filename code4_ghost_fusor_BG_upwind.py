# code4_ghost_fusor_BG_upwind.py
# (4) Code3 + Upwind encoder (replace GCN encoder with upwind forward-difference layers) + plots (A/B)

import os, random, copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader as PyGDataLoader

from error_prop_utils import (
    ensure_reproducibility,
    find_boundary_roots_from_ghost_sources,
    calculate_downstream_distances_multi_source,
    evaluate_epoch_with_nodewise_ghost,
    plot_error_propagation_A,
    build_error_series_per_root,
    plot_and_save_error_propagation_B,
)

# -------- dataset (with ghost) --------
try:
    from dataset_tt_split import LamaHDatasetWithGhosts
except Exception as e:
    raise ImportError("Cannot import LamaHDatasetWithGhosts. Put dataset_tt_split.py in the same folder.") from e

# ---------------- upwind encoder (from your upwind version) ----------------
def create_forward_difference_matrix(num_nodes, edge_index, delta_x, delta_z):
    """
    Sparse forward difference operator D: (D @ u)[src] accumulates outgoing differences to dst.
    Weights are derived from delta_x, delta_z (both positive).
    """
    src = edge_index[0]
    dst = edge_index[1]
    E = edge_index.size(1)
    delta_x = delta_x.view(E, 1)
    delta_z = delta_z.view(E, 1)

    denom = (delta_x.abs() + delta_z.abs()).clamp_min(1e-6)
    w = (delta_z / denom).view(-1)  # [E]

    row = torch.cat([src, src], dim=0)
    col = torch.cat([dst, src], dim=0)
    val = torch.cat([w, -w], dim=0)
    D = torch.sparse_coo_tensor(torch.stack([row, col], dim=0), val, (num_nodes, num_nodes)).coalesce()
    return D

class EdgeMLP(nn.Module):
    def __init__(self, edge_attr_dim, hidden=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(edge_attr_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, 1), nn.Softplus()
        )
    def forward(self, edge_attr):
        return self.net(edge_attr)

class UpwindLayer(nn.Module):
    def __init__(self, hidden_dim, edge_attr_dim, delta_t_param):
        super().__init__()
        self.edge_attr_dim = int(edge_attr_dim)
        self.delta_t = delta_t_param
        self.W = nn.Parameter(torch.empty(hidden_dim, hidden_dim))
        nn.init.xavier_uniform_(self.W)
        self.mlp_dx = EdgeMLP(edge_attr_dim, hidden=16)
        self.mlp_dz = EdgeMLP(edge_attr_dim, hidden=16)
    def forward(self, u, edge_index, edge_attr):
        if edge_attr is None:
            edge_attr = torch.zeros(edge_index.size(1), self.edge_attr_dim, device=u.device)
        if edge_attr.dim() == 1:
            edge_attr = edge_attr.view(-1, self.edge_attr_dim)
        if edge_attr.size(1) != self.edge_attr_dim:
            raise ValueError(f"edge_attr_dim mismatch: expected {self.edge_attr_dim}, got {edge_attr.size(1)}")

        dx = self.mlp_dx(edge_attr) + 1e-6
        dz = self.mlp_dz(edge_attr) + 1e-6
        D = create_forward_difference_matrix(u.size(0), edge_index, dx, dz)
        u = torch.tanh(u @ self.W)
        du = torch.sparse.mm(D, u)
        return u + torch.tanh(self.delta_t * du)

class UpwindGNN(nn.Module):
    def __init__(self, in_ch, hidden_dim, num_layers, edge_attr_dim):
        super().__init__()
        self.lin_in = nn.Linear(in_ch, hidden_dim)
        self.delta_t = nn.Parameter(torch.tensor(0.7))
        self.layers = nn.ModuleList([UpwindLayer(hidden_dim, edge_attr_dim, self.delta_t) for _ in range(num_layers)])
    def forward(self, x, edge_index, edge_attr):
        u = self.lin_in(x)
        for layer in self.layers:
            u = layer(u, edge_index, edge_attr)
        return u

# ---------------- shared blocks ----------------
class MLPHead(nn.Module):
    def __init__(self, in_dim, hid, out_dim=1, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hid), nn.LayerNorm(hid), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hid, hid), nn.LayerNorm(hid), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hid, out_dim),
        )
    def forward(self, x):
        return self.net(x)

class GhostFusor(nn.Module):
    def __init__(self, feat_dim, depth_D, hidden=None, dropout=0.1, use_gate=True, use_dx=True):
        super().__init__()
        self.F = int(feat_dim); self.D = int(depth_D); self.use_dx = bool(use_dx)
        h = hidden or max(32, 2 * self.F * (1 + self.D))
        in_dim = self.F * (1 + self.D) + (1 if self.use_dx else 0)
        self.fc1 = nn.Linear(in_dim, h); self.fc2 = nn.Linear(h, self.F)
        self.act = nn.ReLU(); self.do = nn.Dropout(dropout)
        self.use_gate = bool(use_gate)
        if self.use_gate:
            self.gate = nn.Sequential(
                nn.Linear(in_dim, max(8, h // 2)), nn.ReLU(),
                nn.Linear(max(8, h // 2), 1), nn.Sigmoid()
            )
    def forward(self, ghost_feat, downs_concat, dx_bnd=None):
        x = ghost_feat if downs_concat is None else torch.cat([ghost_feat, downs_concat], dim=1)
        if self.use_dx:
            if dx_bnd is None:
                dx_bnd = torch.ones((ghost_feat.size(0), 1), device=ghost_feat.device, dtype=ghost_feat.dtype)
            x = torch.cat([x, dx_bnd], dim=1)
        z = self.do(self.act(self.fc1(x)))
        delta = self.fc2(z)
        if self.use_gate:
            delta = delta * self.gate(x)
        return delta

class RobinBoundaryRefinerClosedForm(nn.Module):
    def __init__(self, num_nodes_template, init_a=1.0, init_b=1.0,
                 init_lamR=1.0, init_lamb=1.0, init_lamd=1.0):
        super().__init__()
        self.num_nodes_template = int(num_nodes_template)
        self.a = nn.Parameter(torch.tensor(float(init_a)))
        self.b = nn.Parameter(torch.tensor(float(init_b)))

        def inv_softplus(x):
            x = float(x)
            return np.log(np.exp(x) - 1.0 + 1e-8)

        self._lamR_raw = nn.Parameter(torch.tensor(inv_softplus(init_lamR), dtype=torch.float32))
        self._lamb_raw = nn.Parameter(torch.tensor(inv_softplus(init_lamb), dtype=torch.float32))
        self._lamd_raw = nn.Parameter(torch.tensor(inv_softplus(init_lamd), dtype=torch.float32))

        self.c_embed = nn.Embedding(self.num_nodes_template, 1)
        nn.init.zeros_(self.c_embed.weight)

    def lambdas(self):
        eps = 1e-8
        lamR = F.softplus(self._lamR_raw) + eps
        lamb = F.softplus(self._lamb_raw) + eps
        lamd = F.softplus(self._lamd_raw) + eps
        return lamR, lamb, lamd

    def forward(self, hg_hat, hb_hat, dx, ghost_local_idx):
        eps = 1e-8
        dx = dx.clamp_min(1e-6)
        a = self.a
        b = self.b
        beta = b / (dx + eps)
        alpha = a - beta
        c = self.c_embed(ghost_local_idx).to(hg_hat.dtype)

        lamR, lamb, lamd = self.lambdas()
        A = lamb + lamR * (alpha * alpha)
        B = lamR * (alpha * beta)
        C = lamd + lamR * (beta * beta)

        rhs1 = lamb * hg_hat + lamR * alpha * c
        rhs2 = lamd * hb_hat + lamR * beta  * c

        denom = A * C - B * B + eps
        hg_star = (C * rhs1 - B * rhs2) / denom
        hb_star = (-B * rhs1 + A * rhs2) / denom
        return hg_star, hb_star

# ---------------- full model (GhostFusor + optional B/G + UpwindGNN) ----------------
class STGNN_GhostFusor_BG_Upwind_AR(nn.Module):
    def __init__(self, num_nodes_total, node_features, hidden_dim, T_in, T_out,
                 upwind_layers=2, edge_attr_dim=1, dropout=0.1, distance_col=0,
                 downstream_concat_depth=2, alpha_boundary=0.5,
                 use_B=True, use_G=True, boundary_relax=0.7, num_bg_cycles=2,
                 writeback_mode="both",
                 lam_grad=1.0, lam_b=0.1, lam_d=2.0):
        super().__init__()
        self.N_total = int(num_nodes_total)
        self.F = int(node_features)
        self.hidden_dim = int(hidden_dim)
        self.T_in = int(T_in)
        self.T_out = int(T_out)
        self.distance_col = int(distance_col)
        self.D = int(max(1, downstream_concat_depth))
        self.alpha_boundary = float(alpha_boundary)

        self.use_B = bool(use_B)
        self.use_G = bool(use_G)
        self.boundary_relax = float(boundary_relax)
        self.num_bg_cycles = int(num_bg_cycles)
        self.writeback_mode = str(writeback_mode)

        self.lam_grad = float(lam_grad)
        self.lam_b = float(lam_b)
        self.lam_d = float(lam_d)

        self.gnn = UpwindGNN(self.F, self.hidden_dim, int(upwind_layers), int(edge_attr_dim))
        self.rnn = nn.GRUCell(self.hidden_dim, self.hidden_dim)
        self.drop = nn.Dropout(dropout)
        self.pred = MLPHead(self.hidden_dim, self.hidden_dim, out_dim=1, dropout=dropout)

        self.ghost_fuser = GhostFusor(self.F, self.D, dropout=dropout, use_gate=True, use_dx=True)
        self.boundary_refiner = RobinBoundaryRefinerClosedForm(num_nodes_template=self.N_total)

    @staticmethod
    def _first_out(edge_index, N):
        src, dst = edge_index[0], edge_index[1]
        perm = torch.argsort(src)
        src_s, dst_s = src[perm], dst[perm]
        first_mask = torch.ones_like(src_s, dtype=torch.bool); first_mask[1:] = src_s[1:] != src_s[:-1]
        first_out = torch.full((N,), -1, dtype=torch.long, device=edge_index.device)
        first_out[src_s[first_mask]] = dst_s[first_mask]
        return first_out

    @staticmethod
    def _first_edge_value_per_src(edge_index, edge_attr_col):
        src = edge_index[0]
        N = int(max(src.max().item(), edge_index[1].max().item())) + 1
        perm = torch.argsort(src)
        src_s = src[perm]
        val_s = edge_attr_col[perm]
        first_mask = torch.ones_like(src_s, dtype=torch.bool); first_mask[1:] = src_s[1:] != src_s[:-1]
        out = torch.full((N,), -1.0, dtype=edge_attr_col.dtype, device=edge_attr_col.device)
        out[src_s[first_mask]] = val_s[first_mask]
        return out

    def _downstream_k(self, first_out, start_idx, K):
        idx_list = []
        cur = start_idx
        for _ in range(K):
            nxt = first_out[cur]
            bad = nxt < 0
            if bad.any():
                nxt = torch.where(bad, cur, nxt)
            idx_list.append(nxt)
            cur = nxt
        return idx_list

    @staticmethod
    def _closed_form_refine_pairs(ub, ud, inv_dx, ub_hat, ud_hat, lam_grad, lam_b, lam_d):
        invdx2 = inv_dx * inv_dx
        a = lam_grad * invdx2 + lam_b
        b = lam_grad * invdx2
        c = lam_grad * invdx2 + lam_d
        rhs_b = lam_b * ub_hat
        rhs_d = lam_d * ud_hat
        denom = a * c - b * b + 1e-8
        ub_new = (c * rhs_b + b * rhs_d) / denom
        ud_new = (a * rhs_d + b * rhs_b) / denom
        return ub_new, ud_new

    def _global_sweep_once_dst_jacobi(self, x_nodes, edge_index, inv_dx_edges, omega=0.8):
        src, dst = edge_index[0], edge_index[1]
        ui = x_nodes[src]
        uj = x_nodes[dst]
        _, uj_new = self._closed_form_refine_pairs(ui, uj, inv_dx_edges, ui, uj, self.lam_grad, self.lam_b, self.lam_d)

        acc = torch.zeros_like(x_nodes)
        cnt = torch.zeros_like(x_nodes)
        acc.index_add_(0, dst, uj_new)
        cnt.index_add_(0, dst, torch.ones_like(uj_new))
        mask = cnt > 0
        x_out = x_nodes.clone()
        x_out[mask] = (1.0 - omega) * x_nodes[mask] + omega * (acc[mask] / cnt[mask])
        return x_out

    def forward(self, data, y_target_for_loss=None, teacher_forcing_ratio=0.0, epoch_idx=None):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        device = x.device
        N = x.size(0)

        real_mask = data.mask.bool().to(device)
        real_idx = torch.where(real_mask)[0]
        ghost_idx = torch.where(~real_mask)[0]
        if ghost_idx.numel() == 0:
            raise RuntimeError("No ghost nodes found.")

        first_out = self._first_out(edge_index, N)
        b_idx = first_out[ghost_idx]
        if (b_idx < 0).any():
            raise RuntimeError("Some ghost nodes have no outgoing edge to boundary.")

        # dx for each ghost (use first outgoing edge length)
        dist_col = edge_attr[:, self.distance_col].clamp_min(1e-6)
        first_dx = self._first_edge_value_per_src(edge_index, dist_col)
        dx_bnd = first_dx[ghost_idx].clamp_min(1e-6).unsqueeze(-1)

        inv_dx_edges = (1.0 / dist_col).unsqueeze(-1) if self.use_G else None

        ghost_local = (ghost_idx % self.N_total).long()
        dK = self._downstream_k(first_out, b_idx, self.D)

        h = torch.zeros(N, self.hidden_dim, device=device)

        # encoder
        hist_x = []
        for t in range(self.T_in):
            x_t = x[:, t, :]

            ghost_feat = x_t[ghost_idx]
            downs = [x_t[idx] for idx in dK]
            downs_concat = torch.cat(downs, dim=1) if len(downs) > 0 else None
            delta_g = self.ghost_fuser(ghost_feat, downs_concat, dx_bnd=dx_bnd)

            x_t = x_t.clone()
            x_t[ghost_idx] = ghost_feat + self.alpha_boundary * delta_g

            g = self.gnn(x_t, edge_index, edge_attr)
            h = self.drop(self.rnn(g, h))
            hist_x.append(x_t.detach())

        # decoder (+ BG cycles)
        cur = hist_x[-1].clone()
        outs_real = []
        relax = float(max(0.05, min(1.0, self.boundary_relax)))

        for t in range(self.T_out):
            g = self.gnn(cur, edge_index, edge_attr)
            h = self.drop(self.rnn(g, h))
            y = self.pred(h)
            nxt = y.clone()

            use_tf = self.training and (y_target_for_loss is not None) and (random.random() < teacher_forcing_ratio)
            if use_tf:
                nxt[real_idx] = y_target_for_loss[:, t].unsqueeze(-1)

            for _cy in range(self.num_bg_cycles):
                if self.use_B:
                    hg_hat = nxt[ghost_idx]
                    hb_hat = nxt[b_idx]
                    hg_star, hb_star = self.boundary_refiner(hg_hat, hb_hat, dx_bnd, ghost_local)
                    hg_new = (1.0 - relax) * hg_hat + relax * hg_star
                    hb_new = (1.0 - relax) * hb_hat + relax * hb_star
                    nxt = nxt.clone()
                    if self.writeback_mode in ("both", "ghost"):
                        nxt[ghost_idx] = hg_new
                    if self.writeback_mode in ("both", "boundary"):
                        nxt[b_idx] = hb_new

                if self.use_G:
                    nxt = self._global_sweep_once_dst_jacobi(nxt, edge_index, inv_dx_edges, omega=0.8)

            outs_real.append(nxt[real_idx])
            cur = nxt

        return torch.cat(outs_real, dim=1)

# ---------------- train ----------------
def train_epoch(model, loader, criterion, optim, device, tf_ratio=0.0):
    model.train()
    tot = 0.0
    for batch in tqdm(loader, desc="Training"):
        batch = batch.to(device)
        optim.zero_grad()
        y = batch.y.squeeze(-1)
        y_hat = model(batch, y_target_for_loss=y, teacher_forcing_ratio=tf_ratio)
        loss = criterion(y_hat, y)
        loss.backward()
        optim.step()
        tot += float(loss.item())
    return tot / max(1, len(loader))

def main():
    hp = {
        "seed": 42,
        "use_cuda": True,
        "save_dir": "runs_code4_ghost_fusor_BG_upwind",
        "data": {
            "path": "./data/LamaH-CE",
            "years_total": [1981, 2017],
            "root_gauge_id": 0,
            "rewire_graph": False,
            "T_in": 36,
            "T_out": 16,
            "stride": 1,
            "normalized": True,
        },
        "model": {
            "hidden_dim": 64,
            "upwind_layers": 2,
            "dropout": 0.1,
            "distance_col": 0,
            "downstream_concat_depth": 2,
            "alpha_boundary": 0.5,
            "use_B": True,
            "use_G": True,
            "boundary_relax": 0.7,
            "num_bg_cycles": 2,
            "writeback_mode": "both",
        },
        "train": {
            "epochs": 30,
            "batch_size": 1,
            "lr": 1e-3,
            "tf_ratio": 0.0,
            "val_split": 0.1,
            "test_split": 0.1,
            "patience": 5,
            "num_workers": 0,
        },
        "analysis": {"skip_distance0": False, "max_roots_in_plot": 8},
    }

    ensure_reproducibility(hp["seed"])
    device = torch.device("cuda" if (torch.cuda.is_available() and hp["use_cuda"]) else "cpu")
    os.makedirs(hp["save_dir"], exist_ok=True)
    print("Device:", device)

    dataset = LamaHDatasetWithGhosts(
        root_dir=hp["data"]["path"],
        years=hp["data"]["years_total"],
        root_gauge_id=hp["data"]["root_gauge_id"],
        rewire_graph=hp["data"]["rewire_graph"],
        input_window_size=hp["data"]["T_in"],
        total_target_window=hp["data"]["T_out"],
        stride_length=hp["data"]["stride"],
        normalized=hp["data"]["normalized"],
    )
    sample = dataset[0]
    N_total = sample.num_nodes
    Fin = sample.x.size(-1)
    T_out = sample.y.size(1)
    edge_attr_dim = 0 if sample.edge_attr is None else int(sample.edge_attr.size(1))
    template_real_node_ids = torch.where(sample.mask.bool())[0].cpu()

    boundary_roots, _ = find_boundary_roots_from_ghost_sources(sample.edge_index, sample.mask)
    dist_multi = calculate_downstream_distances_multi_source(sample.edge_index, N_total, boundary_roots)
    print(f"N_total={N_total}, Fin={Fin}, edge_attr_dim={edge_attr_dim}, T_out={T_out}, boundary_roots#={len(boundary_roots)}")

    n = len(dataset)
    test_size = int(hp["train"]["test_split"] * n)
    val_size = int(hp["train"]["val_split"] * n)
    train_size = n - val_size - test_size
    train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])

    train_loader = PyGDataLoader(train_set, batch_size=hp["train"]["batch_size"], shuffle=True,
                                 drop_last=True, num_workers=hp["train"]["num_workers"])
    val_loader = PyGDataLoader(val_set, batch_size=hp["train"]["batch_size"], shuffle=False,
                               drop_last=True, num_workers=hp["train"]["num_workers"])
    test_loader = PyGDataLoader(test_set, batch_size=hp["train"]["batch_size"], shuffle=False,
                                drop_last=True, num_workers=hp["train"]["num_workers"])

    model = STGNN_GhostFusor_BG_Upwind_AR(
        N_total, Fin, hp["model"]["hidden_dim"], hp["data"]["T_in"], T_out,
        upwind_layers=hp["model"]["upwind_layers"],
        edge_attr_dim=edge_attr_dim,
        dropout=hp["model"]["dropout"],
        distance_col=hp["model"]["distance_col"],
        downstream_concat_depth=hp["model"]["downstream_concat_depth"],
        alpha_boundary=hp["model"]["alpha_boundary"],
        use_B=hp["model"]["use_B"],
        use_G=hp["model"]["use_G"],
        boundary_relax=hp["model"]["boundary_relax"],
        num_bg_cycles=hp["model"]["num_bg_cycles"],
        writeback_mode=hp["model"]["writeback_mode"],
    ).to(device)

    optim = torch.optim.Adam(model.parameters(), lr=hp["train"]["lr"])
    crit = nn.MSELoss()

    best_val = float("inf"); best_state = None; bad = 0
    for ep in range(1, hp["train"]["epochs"] + 1):
        tr = train_epoch(model, train_loader, crit, optim, device, tf_ratio=hp["train"]["tf_ratio"])
        val_mse, val_mae, val_rmse = evaluate_epoch_with_nodewise_ghost(
            model, val_loader, device, template_real_node_ids, node_distances=dist_multi, return_nodewise=False
        )
        print(f"Epoch {ep:03d} | train={tr:.4f} | val_mse={val_mse:.4f} | val_rmse={val_rmse:.4f}")
        if val_mse < best_val:
            best_val = val_mse
            best_state = copy.deepcopy(model.state_dict())
            bad = 0
        else:
            bad += 1
            if bad >= hp["train"]["patience"]:
                print("Early stopping.")
                break
    if best_state is not None:
        model.load_state_dict(best_state)

    test_mse, test_mae, test_rmse, node_mse_avg = evaluate_epoch_with_nodewise_ghost(
        model, test_loader, device, template_real_node_ids, node_distances=dist_multi, return_nodewise=True
    )
    print(f"Test | mse={test_mse:.4f} | rmse={test_rmse:.4f}")

    # -------- plots (A/B) --------
    plot_error_propagation_A(node_mse_avg, dist_multi, save_dir=hp["save_dir"], show=True,
                             skip_distance0=hp["analysis"]["skip_distance0"])
    df_series = build_error_series_per_root(node_mse_avg, sample.edge_index, N_total, boundary_roots,
                                            max_roots=None, skip_distance0=hp["analysis"]["skip_distance0"])
    plot_and_save_error_propagation_B(df_series, save_dir=hp["save_dir"], show=True,
                                      max_roots_in_plot=hp["analysis"]["max_roots_in_plot"])

if __name__ == "__main__":
    main()
