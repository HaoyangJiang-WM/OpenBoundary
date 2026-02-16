# code2_ghost_fusor.py
# (2) Ghost + GhostFusor (no B/G, no upwind) + Error-propagation plots (A/B)

import os, random, copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.nn import GCNConv

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

# ---------------- model blocks ----------------
class SimpleGCNEncoder(nn.Module):
    def __init__(self, in_ch, hid, layers=2, dropout=0.1):
        super().__init__()
        self.lin = nn.Linear(in_ch, hid)
        self.convs = nn.ModuleList([GCNConv(hid, hid) for _ in range(layers)])
        self.dropout = float(dropout)
    def forward(self, x, edge_index):
        h = F.relu(self.lin(x))
        for conv in self.convs:
            h = F.relu(conv(h, edge_index))
            if self.dropout > 0:
                h = F.dropout(h, p=self.dropout, training=self.training)
        return h

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
    """
    concat([ghost, d1..dD, dx]) -> Δghost
    """
    def __init__(self, feat_dim, depth_D, hidden=None, dropout=0.1, use_gate=True, use_dx=True):
        super().__init__()
        self.F = int(feat_dim)
        self.D = int(depth_D)
        self.use_dx = bool(use_dx)
        h = hidden or max(32, 2 * self.F * (1 + self.D))
        in_dim = self.F * (1 + self.D) + (1 if self.use_dx else 0)
        self.fc1 = nn.Linear(in_dim, h)
        self.fc2 = nn.Linear(h, self.F)
        self.act = nn.ReLU()
        self.do = nn.Dropout(dropout)
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
        z = self.act(self.fc1(x))
        z = self.do(z)
        delta = self.fc2(z)
        if self.use_gate:
            delta = delta * self.gate(x)
        return delta

# ---------------- full model ----------------
class STGNN_GhostFusor_AR(nn.Module):
    def __init__(self, num_nodes_total, node_features, hidden_dim, T_in, T_out,
                 gnn_layers=2, dropout=0.1, distance_col=0, downstream_concat_depth=2, alpha_boundary=0.5):
        super().__init__()
        self.N_total = int(num_nodes_total)
        self.F = int(node_features)
        self.hidden_dim = int(hidden_dim)
        self.T_in = int(T_in)
        self.T_out = int(T_out)
        self.distance_col = int(distance_col)
        self.D = int(max(1, downstream_concat_depth))
        self.alpha_boundary = float(alpha_boundary)

        self.gnn = SimpleGCNEncoder(self.F, self.hidden_dim, layers=gnn_layers, dropout=dropout)
        self.rnn = nn.GRUCell(self.hidden_dim, self.hidden_dim)
        self.drop = nn.Dropout(dropout)
        self.pred = MLPHead(self.hidden_dim, self.hidden_dim, out_dim=1, dropout=dropout)

        self.ghost_fuser = GhostFusor(self.F, self.D, dropout=dropout, use_gate=True, use_dx=True)

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
        first_mask = torch.ones_like(src_s, dtype=torch.bool)
        first_mask[1:] = src_s[1:] != src_s[:-1]
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

    def forward(self, data, y_target_for_loss=None, teacher_forcing_ratio=0.0, epoch_idx=None):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        device = x.device
        N = x.size(0)

        # real/ghost indices
        real_mask = data.mask.bool().to(device)
        real_idx = torch.where(real_mask)[0]
        ghost_idx = torch.where(~real_mask)[0]
        if ghost_idx.numel() == 0:
            raise RuntimeError("No ghost nodes found.")

        # ghost->boundary and dx
        first_out = self._first_out(edge_index, N)
        b_idx = first_out[ghost_idx]
        if (b_idx < 0).any():
            raise RuntimeError("Some ghost nodes have no outgoing edge to boundary.")

        dist_col = edge_attr[:, self.distance_col].clamp_min(1e-6)
        first_dx = self._first_edge_value_per_src(edge_index, dist_col)
        dx_bnd = first_dx[ghost_idx].clamp_min(1e-6).unsqueeze(-1)

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

            g = self.gnn(x_t, edge_index)
            h = self.drop(self.rnn(g, h))
            hist_x.append(x_t.detach())

        # decoder
        cur = hist_x[-1].clone()
        outs_real = []
        for t in range(self.T_out):
            g = self.gnn(cur, edge_index)
            h = self.drop(self.rnn(g, h))
            y = self.pred(h)  # [N,1]
            nxt = y.clone()

            # TF on real nodes only
            use_tf = self.training and (y_target_for_loss is not None) and (random.random() < teacher_forcing_ratio)
            if use_tf:
                nxt[real_idx] = y_target_for_loss[:, t].unsqueeze(-1)

            outs_real.append(nxt[real_idx])
            cur = nxt

        return torch.cat(outs_real, dim=1)  # [B*N_real_template, T_out]

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
        "save_dir": "runs_code2_ghost_fusor",
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
            "gnn_layers": 2,
            "dropout": 0.1,
            "distance_col": 0,
            "downstream_concat_depth": 2,
            "alpha_boundary": 0.5,
        },
        "train": {
            "epochs": 30,
            "batch_size": 1,  # 强烈建议 1
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
    template_real_node_ids = torch.where(sample.mask.bool())[0].cpu()

    boundary_roots, _ = find_boundary_roots_from_ghost_sources(sample.edge_index, sample.mask)
    dist_multi = calculate_downstream_distances_multi_source(sample.edge_index, N_total, boundary_roots)
    print(f"N_total={N_total}, Fin={Fin}, T_out={T_out}, boundary_roots#={len(boundary_roots)}")

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

    model = STGNN_GhostFusor_AR(
        N_total, Fin, hp["model"]["hidden_dim"], hp["data"]["T_in"], T_out,
        gnn_layers=hp["model"]["gnn_layers"],
        dropout=hp["model"]["dropout"],
        distance_col=hp["model"]["distance_col"],
        downstream_concat_depth=hp["model"]["downstream_concat_depth"],
        alpha_boundary=hp["model"]["alpha_boundary"],
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
