# code1_noghost_original.py
# (1) No ghost: original STGNN-AR + Error-propagation plots (A/B)

import os, copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.nn import GCNConv

from error_prop_utils import (
    ensure_reproducibility,
    find_source_nodes,
    calculate_downstream_distances_multi_source,
    evaluate_epoch_no_ghost,
    plot_error_propagation_A,
    build_error_series_per_root,
    plot_and_save_error_propagation_B,
)

# -------- dataset (no ghost) --------
try:
    from dataset_tt import LamaHDataset
except Exception as e:
    raise ImportError("Cannot import LamaHDataset. Put dataset_tt.py in the same folder.") from e

# ---------------- model ----------------
class GCNEncoder(nn.Module):
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

class STGNN_AR(nn.Module):
    def __init__(self, num_nodes, node_features, hidden_dim, T_in, T_out, gnn_layers=2, dropout=0.1):
        super().__init__()
        self.num_nodes = int(num_nodes)
        self.T_in = int(T_in)
        self.T_out = int(T_out)
        self.gnn = GCNEncoder(node_features, hidden_dim, layers=gnn_layers, dropout=dropout)
        self.rnn = nn.GRUCell(hidden_dim, hidden_dim)
        self.drop = nn.Dropout(dropout)
        self.pred = MLPHead(hidden_dim, hidden_dim, out_dim=1, dropout=dropout)

    def forward(self, data, teacher_forcing_ratio=0.0):
        x, edge_index = data.x, data.edge_index
        device = x.device
        N = x.size(0)
        h = torch.zeros(N, self.rnn.hidden_size, device=device)

        hist_x = []
        for t in range(self.T_in):
            x_t = x[:, t, :]
            g = self.gnn(x_t, edge_index)
            h = self.drop(self.rnn(g, h))
            hist_x.append(x_t.detach())

        cur = hist_x[-1].clone()
        outs = []
        for t in range(self.T_out):
            g = self.gnn(cur, edge_index)
            h = self.drop(self.rnn(g, h))
            y = self.pred(h)     # [N,1]
            outs.append(y)
            cur = y
        return torch.cat(outs, dim=1)  # [N, T_out] (batch_size=1 assumed)

# ---------------- train ----------------
def train_epoch(model, loader, criterion, optim, device):
    model.train()
    tot = 0.0
    for batch in tqdm(loader, desc="Training"):
        batch = batch.to(device)
        optim.zero_grad()
        y = batch.y.squeeze(-1)                # [B*N, T]
        y_hat = model(batch, teacher_forcing_ratio=0.0)
        loss = criterion(y_hat, y)
        loss.backward()
        optim.step()
        tot += float(loss.item())
    return tot / max(1, len(loader))

def main():
    hp = {
        "seed": 42,
        "use_cuda": True,
        "save_dir": "runs_code1_noghost",
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
        "model": {"hidden_dim": 64, "gnn_layers": 2, "dropout": 0.1},
        "train": {
            "epochs": 30,
            "batch_size": 1,   # 推荐 1，保证 node-wise 对齐
            "lr": 1e-3,
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

    dataset = LamaHDataset(
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
    N = sample.num_nodes
    Fin = sample.x.size(-1)
    T_out = sample.y.size(1)
    print(f"N={N}, Fin={Fin}, T_out={T_out}")

    root_ids = find_source_nodes(sample.edge_index, N)
    dist_multi = calculate_downstream_distances_multi_source(sample.edge_index, N, root_ids)

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

    model = STGNN_AR(N, Fin, hp["model"]["hidden_dim"], hp["data"]["T_in"], T_out,
                     gnn_layers=hp["model"]["gnn_layers"], dropout=hp["model"]["dropout"]).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=hp["train"]["lr"])
    crit = nn.MSELoss()

    best_val = float("inf"); best_state = None; bad = 0
    for ep in range(1, hp["train"]["epochs"] + 1):
        tr = train_epoch(model, train_loader, crit, optim, device)
        val_mse, val_mae, val_rmse = evaluate_epoch_no_ghost(model, val_loader, device, num_nodes=N,
                                                             node_distances=dist_multi, return_nodewise=False)
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

    test_mse, test_mae, test_rmse, node_mse_avg = evaluate_epoch_no_ghost(
        model, test_loader, device, num_nodes=N, node_distances=dist_multi, return_nodewise=True
    )
    print(f"Test | mse={test_mse:.4f} | rmse={test_rmse:.4f}")

    # -------- plots (A/B) --------
    plot_error_propagation_A(node_mse_avg, dist_multi, save_dir=hp["save_dir"], show=True,
                             skip_distance0=hp["analysis"]["skip_distance0"])
    df_series = build_error_series_per_root(node_mse_avg, sample.edge_index, N, root_ids,
                                            max_roots=None, skip_distance0=hp["analysis"]["skip_distance0"])
    plot_and_save_error_propagation_B(df_series, save_dir=hp["save_dir"], show=True,
                                      max_roots_in_plot=hp["analysis"]["max_roots_in_plot"])

if __name__ == "__main__":
    main()
