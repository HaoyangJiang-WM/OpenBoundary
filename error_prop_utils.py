# error_prop_utils.py
# Shared utilities for error-propagation analysis (A/B plots) for LamaH graphs (with or without ghosts).

import os, random, collections
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

def ensure_reproducibility(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# ---------------- graph helpers ----------------
def find_source_nodes(edge_index: torch.Tensor, num_nodes: int):
    dst = edge_index[1]
    indeg = torch.bincount(dst, minlength=num_nodes)
    return (indeg == 0).nonzero(as_tuple=False).view(-1).tolist()

def find_in_degree0_nodes(edge_index: torch.Tensor, num_nodes: int):
    return find_source_nodes(edge_index, num_nodes)

def first_out_dst(edge_index: torch.Tensor, num_nodes: int):
    """first_out[src] = first dst for each src (or -1)."""
    src, dst = edge_index[0], edge_index[1]
    perm = torch.argsort(src)
    src_s, dst_s = src[perm], dst[perm]
    first_mask = torch.ones_like(src_s, dtype=torch.bool)
    first_mask[1:] = src_s[1:] != src_s[:-1]
    out = torch.full((num_nodes,), -1, dtype=torch.long, device=edge_index.device)
    out[src_s[first_mask]] = dst_s[first_mask]
    return out

def calculate_downstream_distances(edge_index: torch.Tensor, num_nodes: int, root_id: int):
    forward_adj = collections.defaultdict(list)
    E = edge_index.size(1)
    for i in range(E):
        s = int(edge_index[0, i]); d = int(edge_index[1, i])
        forward_adj[s].append(d)

    dist = {i: float("inf") for i in range(num_nodes)}
    root_id = int(root_id)
    if root_id < 0 or root_id >= num_nodes:
        return dist

    q = collections.deque([root_id])
    dist[root_id] = 0
    while q:
        u = q.popleft()
        for v in forward_adj.get(u, []):
            if dist[v] == float("inf"):
                dist[v] = dist[u] + 1
                q.append(v)
    return dist

def calculate_downstream_distances_multi_source(edge_index: torch.Tensor, num_nodes: int, root_ids):
    forward_adj = collections.defaultdict(list)
    E = edge_index.size(1)
    for i in range(E):
        s = int(edge_index[0, i]); d = int(edge_index[1, i])
        forward_adj[s].append(d)

    dist = {i: float("inf") for i in range(num_nodes)}
    q = collections.deque()
    for r in root_ids:
        r = int(r)
        if 0 <= r < num_nodes and dist[r] == float("inf"):
            dist[r] = 0
            q.append(r)

    while q:
        u = q.popleft()
        for v in forward_adj.get(u, []):
            if dist[v] == float("inf"):
                dist[v] = dist[u] + 1
                q.append(v)
    return dist

def find_boundary_roots_from_ghost_sources(edge_index: torch.Tensor, mask_real: torch.Tensor):
    """
    Ghost graph convention:
      - mask_real=True for real nodes; False for ghost nodes
      - each ghost has one outgoing edge to its boundary real node
    Return:
      boundary_roots: list[int]  (real nodes that receive ghost forcing)
      ghost_sources: list[int]   (ghost nodes with in-degree 0)
    """
    num_nodes = int(mask_real.numel())
    mask_real = mask_real.bool()
    ghost_mask = ~mask_real

    indeg0 = find_in_degree0_nodes(edge_index, num_nodes)
    ghost_sources = [nid for nid in indeg0 if bool(ghost_mask[nid].item())]

    fo = first_out_dst(edge_index, num_nodes)
    boundary_roots = []
    for g in ghost_sources:
        b = int(fo[g].item())
        if 0 <= b < num_nodes and bool(mask_real[b].item()):
            boundary_roots.append(b)
    boundary_roots = sorted(list(set(boundary_roots)))

    if len(boundary_roots) == 0:
        # fallback: any real node with a ghost incoming edge
        src, dst = edge_index[0], edge_index[1]
        for i in range(edge_index.size(1)):
            s = int(src[i].item()); d = int(dst[i].item())
            if bool(ghost_mask[s].item()) and bool(mask_real[d].item()):
                boundary_roots.append(d)
        boundary_roots = sorted(list(set(boundary_roots)))

    return boundary_roots, ghost_sources

# ---------------- evaluation ----------------
@torch.no_grad()
def evaluate_epoch_no_ghost(model, loader, device, num_nodes, node_distances=None, return_nodewise=False):
    model.eval()
    total_se = 0.0
    total_ae = 0.0
    total_count = 0

    node_se_sum = collections.defaultdict(float)
    node_count = collections.defaultdict(int)

    for batch in tqdm(loader, desc="Evaluating"):
        batch = batch.to(device)
        y = batch.y.squeeze(-1)  # [B*N, T]
        y_hat = model(batch, teacher_forcing_ratio=0.0)

        se = (y_hat - y) ** 2
        ae = (y_hat - y).abs()
        total_se += float(se.sum().item())
        total_ae += float(ae.sum().item())
        total_count += int(se.numel())

        if return_nodewise:
            mse_per_row = se.mean(dim=1)  # [B*N]
            for flat_idx in range(mse_per_row.numel()):
                nid = int(flat_idx % num_nodes)
                if node_distances is not None and node_distances.get(nid, float("inf")) == float("inf"):
                    continue
                node_se_sum[nid] += float(mse_per_row[flat_idx].item())
                node_count[nid] += 1

    avg_mse = total_se / max(1, total_count)
    avg_mae = total_ae / max(1, total_count)
    avg_rmse = float(np.sqrt(avg_mse))

    if not return_nodewise:
        return avg_mse, avg_mae, avg_rmse

    node_mse_avg = {nid: node_se_sum[nid] / node_count[nid] for nid in node_se_sum if node_count[nid] > 0}
    return avg_mse, avg_mae, avg_rmse, node_mse_avg

@torch.no_grad()
def evaluate_epoch_with_nodewise_ghost(model, loader, device, template_real_node_ids, node_distances=None, return_nodewise=False):
    """
    Ghost models return y_pred_real with shape [B*N_real_template, T_out].
    We map flat rows back to template node ids via modulo.
    """
    model.eval()
    total_se = 0.0
    total_ae = 0.0
    total_count = 0

    node_se_sum = collections.defaultdict(float)
    node_count = collections.defaultdict(int)

    template_real_node_ids = torch.as_tensor(template_real_node_ids, dtype=torch.long)
    N_real_template = int(template_real_node_ids.numel())

    for batch in tqdm(loader, desc="Evaluating"):
        batch = batch.to(device)
        y = batch.y.squeeze(-1)
        y_hat = model(batch, y_target_for_loss=None, teacher_forcing_ratio=0.0, epoch_idx=None)

        se = (y_hat - y) ** 2
        ae = (y_hat - y).abs()
        total_se += float(se.sum().item())
        total_ae += float(ae.sum().item())
        total_count += int(se.numel())

        if return_nodewise:
            mse_per_row = se.mean(dim=1)
            for flat_idx in range(mse_per_row.numel()):
                local_pos = int(flat_idx % N_real_template)
                nid = int(template_real_node_ids[local_pos].item())
                if node_distances is not None and node_distances.get(nid, float("inf")) == float("inf"):
                    continue
                node_se_sum[nid] += float(mse_per_row[flat_idx].item())
                node_count[nid] += 1

    avg_mse = total_se / max(1, total_count)
    avg_mae = total_ae / max(1, total_count)
    avg_rmse = float(np.sqrt(avg_mse))

    if not return_nodewise:
        return avg_mse, avg_mae, avg_rmse

    node_mse_avg = {nid: node_se_sum[nid] / node_count[nid] for nid in node_se_sum if node_count[nid] > 0}
    return avg_mse, avg_mae, avg_rmse, node_mse_avg

# ---------------- plotting (A/B) ----------------
def plot_error_propagation_A(node_mse_avg, node_distances_multi, save_dir=None, show=True, skip_distance0=False):
    items = []
    for nid, mse in node_mse_avg.items():
        d = node_distances_multi.get(int(nid), float("inf"))
        if d == float("inf"):
            continue
        if skip_distance0 and int(d) == 0:
            continue
        items.append((int(d), int(nid), float(mse)))
    items.sort(key=lambda x: (x[0], x[1]))
    if len(items) == 0:
        print("[A] No reachable nodes to plot.")
        return

    xs = list(range(len(items)))
    ys = [m for _, _, m in items]

    plt.figure(figsize=(10, 5))
    plt.plot(xs, ys, marker="o", linewidth=2, label="Node-wise MSE")
    plt.title("Error Propagation (A)")
    plt.xlabel("Index (Upstream → Downstream)")
    plt.ylabel("MSE")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, "error_propagation_A_line.png")
        plt.savefig(path, dpi=200, bbox_inches="tight")
        print(f"[A] Saved: {path}")

    if show: plt.show()
    else: plt.close()

def build_error_series_per_root(node_mse_avg, edge_index, num_nodes, root_ids, max_roots=None, skip_distance0=False):
    rows = []
    roots = root_ids if max_roots is None else root_ids[:max_roots]
    for root in roots:
        dist = calculate_downstream_distances(edge_index, num_nodes, int(root))
        by_d = collections.defaultdict(list)
        for nid, mse in node_mse_avg.items():
            d = dist.get(int(nid), float("inf"))
            if d == float("inf"):
                continue
            if skip_distance0 and int(d) == 0:
                continue
            by_d[int(d)].append(float(mse))
        if not by_d:
            continue
        for idx, d in enumerate(sorted(by_d.keys())):
            vals = by_d[d]
            rows.append({
                "root_id": int(root),
                "distance": int(d),
                "index": int(idx),
                "mean_mse": float(np.mean(vals)),
                "num_nodes": int(len(vals)),
            })
    return pd.DataFrame(rows)

def plot_and_save_error_propagation_B(df_series, save_dir=None, show=True, max_roots_in_plot=8):
    if df_series is None or len(df_series) == 0:
        print("[B] Empty df_series.")
        return

    plt.figure(figsize=(10, 5))
    roots = sorted(df_series["root_id"].unique().tolist())[:max_roots_in_plot]
    for root in roots:
        sub = df_series[df_series["root_id"] == root].sort_values("index")
        plt.plot(sub["index"].values, sub["mean_mse"].values, marker="o", linewidth=2, label=f"root {root}")
    plt.title("Error Propagation (B): per-root curves")
    plt.xlabel("Index (distance order)")
    plt.ylabel("Mean MSE")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        fig_path = os.path.join(save_dir, "error_propagation_B_multi_roots.png")
        plt.savefig(fig_path, dpi=200, bbox_inches="tight")
        print(f"[B] Saved: {fig_path}")

        csv_path = os.path.join(save_dir, "error_propagation_B_series.csv")
        pkl_path = os.path.join(save_dir, "error_propagation_B_series.pkl")
        df_series.to_csv(csv_path, index=False)
        df_series.to_pickle(pkl_path)
        print(f"[B] Saved: {csv_path}")
        print(f"[B] Saved: {pkl_path}")

    if show: plt.show()
    else: plt.close()
