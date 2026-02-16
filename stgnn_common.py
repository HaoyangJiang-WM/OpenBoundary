# stgnn_common.py
# -*- coding: utf-8 -*-

import os
import math
import copy
import random
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, Callable, List

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import random_split
from torch_geometric.nn import GCNConv
from torch_geometric.loader import DataLoader as PyGDataLoader

# -------- dataset import (与你现有代码一致) --------
try:
    from dataset_tt import LamaHDataset
except ImportError:
    raise RuntimeError("Error: Could not import LamaHDataset (dataset_tt.py).")

# -------- safe scatter_add fallback --------
try:
    from torch_scatter import scatter_add
except Exception:
    def scatter_add(src, index, dim=0, dim_size=None):
        out = torch.zeros(dim_size, device=src.device, dtype=src.dtype)
        out.index_add_(0, index, src)
        return out


# ==========================
# Utils
# ==========================
def ensure_reproducibility(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # 你的环境里通常更希望 benchmark=True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    print(f"[Seed] {seed}")


def metric_batch(preds: torch.Tensor, labels: torch.Tensor) -> Tuple[float, float]:
    # preds/labels: [N, T]
    if preds.shape != labels.shape:
        raise ValueError(f"Shape mismatch: preds {preds.shape}, labels {labels.shape}")
    mae_val = torch.mean(torch.abs(preds - labels)).item()
    mse_val = torch.mean((preds - labels) ** 2).item()
    return mse_val, mae_val


def make_dir(p: str):
    os.makedirs(p, exist_ok=True)


def boundary_and_downstream_indices(edge_index: torch.Tensor, N: int):
    """
    boundary: indeg==0
    down: 取 boundary 每个 src 的“第一条出边” dst（没有则回退本身）
    """
    src, dst = edge_index[0], edge_index[1]
    indeg = torch.bincount(dst, minlength=N)
    bnd = (indeg == 0).nonzero(as_tuple=False).view(-1)
    if bnd.numel() == 0:
        return bnd, bnd

    perm = torch.argsort(src)
    src_s, dst_s = src[perm], dst[perm]
    first_mask = torch.ones_like(src_s, dtype=torch.bool)
    first_mask[1:] = src_s[1:] != src_s[:-1]

    first_out = torch.full((N,), -1, dtype=torch.long, device=edge_index.device)
    first_out[src_s[first_mask]] = dst_s[first_mask]
    down = first_out[bnd]
    down = torch.where(down < 0, bnd, down)
    return bnd, down


def first_edge_value_per_src(edge_index: torch.Tensor, edge_val: torch.Tensor, N: int):
    """取每个 src 的第一条出边的 edge_val（无则 -1）"""
    src = edge_index[0]
    perm = torch.argsort(src)
    src_s = src[perm]
    val_s = edge_val[perm]
    first_mask = torch.ones_like(src_s, dtype=torch.bool)
    first_mask[1:] = src_s[1:] != src_s[:-1]
    out = torch.full((N,), -1.0, dtype=edge_val.dtype, device=edge_val.device)
    out[src_s[first_mask]] = val_s[first_mask]
    return out


# ==========================
# Basic Blocks
# ==========================
class GNNLayer(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.1):
        super().__init__()
        self.conv = GCNConv(in_channels, out_channels)
        self.norm = nn.LayerNorm(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ReLU()

    def forward(self, x, edge_index, edge_weight=None):
        x = self.conv(x, edge_index, edge_weight)
        x = self.norm(x)
        x = self.act(x)
        x = self.dropout(x)
        return x


class RNNBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, rnn_type='gru'):
        super().__init__()
        rnn_type = rnn_type.lower()
        self.is_lstm = (rnn_type == 'lstm')
        if self.is_lstm:
            self.rnn_cell = nn.LSTMCell(input_dim, hidden_dim)
        elif rnn_type == 'rnn':
            self.rnn_cell = nn.RNNCell(input_dim, hidden_dim)
        else:
            self.rnn_cell = nn.GRUCell(input_dim, hidden_dim)

    def forward(self, x_step, h_prev):
        if self.is_lstm:
            return self.rnn_cell(x_step, h_prev)
        else:
            return self.rnn_cell(x_step, h_prev)


class MLPPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=1, num_layers=3, dropout=0.1):
        super().__init__()
        layers = []
        norms = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        norms.append(nn.LayerNorm(hidden_dim))
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            norms.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.layers = nn.ModuleList(layers)
        self.norms = nn.ModuleList(norms)
        self.drop = nn.Dropout(dropout)
        self.act = nn.ReLU()

    def forward(self, x):
        for layer, norm in zip(self.layers[:-1], self.norms):
            x = self.act(norm(layer(x)))
            x = self.drop(x)
        return self.layers[-1](x)


class BoundaryFuser(nn.Module):
    """
    输入: concat(self, down, (self-down)/dx) -> Δb
    外部做: x_b += α * Δb
    """
    def __init__(self, feat_dim, hidden=None, dropout=0.0, use_gate=True):
        super().__init__()
        h = hidden or max(16, 2 * feat_dim)
        in_dim = 3 * feat_dim
        self.fc1 = nn.Linear(in_dim, h)
        self.fc2 = nn.Linear(h, feat_dim)
        self.act = nn.ReLU()
        self.do = nn.Dropout(dropout)
        self.use_gate = use_gate
        if use_gate:
            self.gate = nn.Sequential(
                nn.Linear(in_dim, max(4, h // 2)),
                nn.ReLU(),
                nn.Linear(max(4, h // 2), 1),
                nn.Sigmoid()
            )

    def forward(self, self_feat, down_feat, diff_norm):
        x = torch.cat([self_feat, down_feat, diff_norm], dim=1)
        z = self.act(self.fc1(x))
        z = self.do(z)
        delta = self.fc2(z)
        if self.use_gate:
            g = self.gate(x)
            delta = delta * g
        return delta


# ==========================
# Physics operators (explicit + convex)
# ==========================
def build_inv_dx(edge_attr: torch.Tensor, dx_col: int):
    dx = edge_attr[:, dx_col].clamp_min(1e-6)
    return 1.0 / dx


def build_slope(edge_attr: torch.Tensor, dx_col: int, dz_col: Optional[int] = None, slope_col: Optional[int] = None):
    if slope_col is not None and slope_col < edge_attr.size(1):
        return edge_attr[:, slope_col]
    if dz_col is not None and dz_col < edge_attr.size(1):
        dx = edge_attr[:, dx_col].clamp_min(1e-6)
        dz = edge_attr[:, dz_col]
        return dz / dx
    return None


def slope_term_nodes(edge_index: torch.Tensor, slope_e: Optional[torch.Tensor], N: int, like: torch.Tensor):
    if slope_e is None:
        return torch.zeros_like(like)
    # 累加到 dst
    dst = edge_index[1]
    s = slope_e.view(-1, 1)
    out = torch.zeros_like(like)
    out.index_add_(0, dst, s.expand(-1, like.size(1)))
    return out


def D1_upwind(x: torch.Tensor, edge_index: torch.Tensor, inv_dx_e: torch.Tensor):
    """
    i=dst, j=src, out[i] += (x[i]-x[j])*inv_dx
    """
    src, dst = edge_index[0], edge_index[1]
    inv = inv_dx_e.view(-1, 1)
    diff = (x[dst] - x[src]) * inv
    out = torch.zeros_like(x)
    out.index_add_(0, dst, diff)
    return out


def D1_symmetric(x: torch.Tensor, edge_index: torch.Tensor, inv_dx_e: torch.Tensor):
    """
    “中心/对称”版本：out[dst]+=diff, out[src]+=-diff
    """
    src, dst = edge_index[0], edge_index[1]
    inv = inv_dx_e.view(-1, 1)
    diff = (x[dst] - x[src]) * inv
    out = torch.zeros_like(x)
    out.index_add_(0, dst, diff)
    out.index_add_(0, src, -diff)
    return out


class PhysicsExplicitGStep(nn.Module):
    """
    显式一步（G）：u_{t+1} = u_t - dt * ( u_t * D1(u_t) + g * slope_nodes )
    其中 D1 可选 symmetric 或 upwind
    """
    def __init__(self, dt_init=0.2, dt_min=0.01, dt_max=2.0, g_init=9.81, mode="symmetric"):
        super().__init__()
        self.dt = nn.Parameter(torch.tensor(float(dt_init)))
        self.dt_min = float(dt_min)
        self.dt_max = float(dt_max)
        self.g = nn.Parameter(torch.tensor(float(g_init)))
        assert mode in ["symmetric", "upwind"]
        self.mode = mode

    def forward(self, u: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor,
                dx_col=0, dz_col=1, slope_col=None):
        N = u.size(0)
        inv_dx = build_inv_dx(edge_attr, dx_col)
        slope_e = build_slope(edge_attr, dx_col, dz_col=dz_col, slope_col=slope_col)
        S = slope_term_nodes(edge_index, slope_e, N, like=u)
        dt = torch.clamp(self.dt, self.dt_min, self.dt_max)

        if self.mode == "upwind":
            d1 = D1_upwind(u, edge_index, inv_dx)
        else:
            d1 = D1_symmetric(u, edge_index, inv_dx)

        u_next = u - dt * (u * d1 + self.g * S)
        return u_next


class PhysicsLSGStep(nn.Module):
    """
    你给的那份代码里的 “线性化—凸LS (semi-implicit) refiner” 的精简版：
      A(u_t) u_{t+1} = b(u_t)
    A = I + dt * diag(u_t) D1_upwind(·)     (这里固定用 upwind D1 做 A)
    b = u_t - dt*g*slope_nodes
    求解器：fs (DAG forward sweep) / jacobi / bicgstab / necg(正规方程CG)
    """
    def __init__(self,
                 lambda_reg=0.0,
                 dt_init=0.3, dt_min=0.02, dt_max=2.0,
                 cg_max_iter=30, cg_tol=1e-4,
                 solver="necg",
                 jacobi_sweeps=2, jacobi_omega=0.95,
                 krylov_max_iter=50, krylov_tol=1e-6):
        super().__init__()
        self.lambda_reg = float(lambda_reg)
        self.dt = nn.Parameter(torch.tensor(float(dt_init)))
        self.dt_min = float(dt_min)
        self.dt_max = float(dt_max)
        self.g_hat = nn.Parameter(torch.tensor(9.81))
        self.cg_max_iter = int(cg_max_iter)
        self.cg_tol = float(cg_tol)
        self.solver = solver.lower()
        self.jacobi_sweeps = int(jacobi_sweeps)
        self.jacobi_omega = float(jacobi_omega)
        self.krylov_max_iter = int(krylov_max_iter)
        self.krylov_tol = float(krylov_tol)

        # cache
        self._built_sig = None
        self.inv_dx = None
        self.slope_e = None
        self._sum_w = None
        self._layers = None
        self._edges_by_lvl_ptr = None
        self._src_sorted = None
        self._dst_sorted = None
        self._w_sorted = None

    def _ensure_built(self, N: int, edge_index: torch.Tensor, edge_attr: torch.Tensor,
                      dx_col=0, dz_col=1, slope_col=None):
        sig = (int(N), int(edge_index.size(1)))
        if self._built_sig == sig and self.inv_dx is not None:
            return

        self.inv_dx = build_inv_dx(edge_attr, dx_col)
        self.slope_e = build_slope(edge_attr, dx_col, dz_col=dz_col, slope_col=slope_col)

        src, dst = edge_index[0], edge_index[1]
        sum_w = torch.zeros(N, device=edge_index.device)
        sum_w.index_add_(0, dst, self.inv_dx)
        self._sum_w = sum_w.view(-1, 1)

        # topo layering (DAG 必须)
        indeg = torch.bincount(dst, minlength=N).clone()
        q0 = (indeg == 0).nonzero(as_tuple=False).view(-1)
        level = -torch.ones(N, dtype=torch.long, device=edge_index.device)
        level[q0] = 0

        perm_by_src = torch.argsort(src)
        src_s, dst_s = src[perm_by_src], dst[perm_by_src]
        cnt = torch.bincount(src_s, minlength=N)
        off = torch.zeros(N + 1, dtype=torch.long, device=edge_index.device)
        off[1:] = torch.cumsum(cnt, 0)

        from collections import deque
        dq = deque(q0.tolist())
        while dq:
            i = dq.popleft()
            s, e = off[i].item(), off[i + 1].item()
            if e > s:
                ks = dst_s[s:e]
                new_lvl = level[i].item() + 1
                mask = (level[ks] < new_lvl)
                if mask.any():
                    upd = ks[mask]
                    level[upd] = new_lvl
                    dq.extend(upd.tolist())

        if (level < 0).any():
            raise RuntimeError("Graph has cycles; convex refiner requires DAG topo ordering.")

        L = int(level.max().item())
        self._layers = [(level == l).nonzero(as_tuple=False).view(-1) for l in range(L + 1)]

        # edges sorted by src level
        src_lvl = level[src]
        order = torch.argsort(src_lvl)
        counts = torch.bincount(src_lvl, minlength=L + 1)
        ptr = torch.zeros(L + 2, dtype=torch.long, device=edge_index.device)
        ptr[1:] = torch.cumsum(counts, 0)

        self._edges_by_lvl_ptr = ptr
        self._src_sorted = src[order]
        self._dst_sorted = dst[order]
        self._w_sorted = self.inv_dx[order].view(-1, 1)

        self._built_sig = sig

    def D1(self, x: torch.Tensor, edge_index: torch.Tensor):
        # upwind D1
        return D1_upwind(x, edge_index, self.inv_dx)

    def D1_T(self, y: torch.Tensor, edge_index: torch.Tensor):
        # 对应上面 D1 的转置（用于正规方程）
        src, dst = edge_index[0], edge_index[1]
        inv = self.inv_dx.view(-1, 1)
        yd = y[dst] * inv
        out = torch.zeros_like(y)
        out.index_add_(0, dst, yd)
        out.index_add_(0, src, -yd)
        return out

    def A_mv(self, v: torch.Tensor, u_t: torch.Tensor, edge_index: torch.Tensor, dt_eff: torch.Tensor):
        return v + dt_eff * (u_t * self.D1(v, edge_index))

    def AT_mv(self, y: torch.Tensor, u_t: torch.Tensor, edge_index: torch.Tensor, dt_eff: torch.Tensor):
        return y + dt_eff * self.D1_T(u_t * y, edge_index)

    def laplacian(self, v: torch.Tensor, edge_index: torch.Tensor):
        # 简单正则项（对称）
        src, dst = edge_index[0], edge_index[1]
        w = self.inv_dx.view(-1, 1)
        diff = (v[dst] - v[src]) * w
        out = torch.zeros_like(v)
        out.index_add_(0, dst, diff)
        out.index_add_(0, src, -diff)
        return out

    def _diag_prec_A(self, u_t: torch.Tensor, dt_eff: torch.Tensor):
        diag = 1.0 + dt_eff * (u_t.squeeze(-1) * self._sum_w.squeeze(-1))
        return 1.0 / diag.clamp_min(1e-8)

    def _solve_fs(self, u_now: torch.Tensor, b: torch.Tensor, dt_eff: torch.Tensor):
        layers = self._layers
        sum_w = self._sum_w
        src_s = self._src_sorted
        dst_s = self._dst_sorted
        w_s = self._w_sorted
        ptr = self._edges_by_lvl_ptr

        accum = torch.zeros_like(u_now)
        u_next = torch.zeros_like(u_now)

        for l in range(len(layers)):
            idx = layers[l]
            if idx.numel() > 0:
                alpha = 1.0 + dt_eff * (u_now[idx] * sum_w[idx])
                rhs = b[idx] + dt_eff * (u_now[idx] * accum[idx])
                u_next[idx] = rhs / alpha.clamp_min(1e-8)

            s, e = ptr[l].item(), ptr[l + 1].item()
            if e > s:
                src_e = src_s[s:e]
                dst_e = dst_s[s:e]
                add = w_s[s:e] * u_next[src_e]
                accum.index_add_(0, dst_e, add)
        return u_next

    def _solve_jacobi(self, u0: torch.Tensor, u_now: torch.Tensor, b: torch.Tensor, dt_eff: torch.Tensor):
        src, dst = self._src_sorted, self._dst_sorted  # 用原顺序也行，这里无所谓
        w = self.inv_dx.view(-1, 1)
        denom = (1.0 + dt_eff * (u_now * self._sum_w)).clamp_min(1e-8)
        u = u0.clone()
        for _ in range(max(1, self.jacobi_sweeps)):
            contrib = torch.zeros_like(u)
            contrib.index_add_(0, dst, w * u[src])
            u_new = (b + dt_eff * (u_now * contrib)) / denom
            u = (1.0 - self.jacobi_omega) * u + self.jacobi_omega * u_new
        return u

    def _bicgstab(self, mv, rhs, x0=None, max_iter=50, tol=1e-6, Minv=None):
        x = torch.zeros_like(rhs) if x0 is None else x0.clone()
        r = rhs - mv(x)
        r_hat = r.clone()
        rho = torch.tensor(1.0, device=rhs.device)
        alpha = torch.tensor(1.0, device=rhs.device)
        omega = torch.tensor(1.0, device=rhs.device)
        v = torch.zeros_like(rhs)
        p = torch.zeros_like(rhs)

        def M_apply(z):
            return z if Minv is None else z * Minv.view(-1, 1)

        if torch.norm(r) < tol:
            return x

        for _ in range(max_iter):
            rho_new = torch.sum(r_hat * r)
            if rho_new.abs() < 1e-30:
                break
            beta = (rho_new / rho) * (alpha / (omega + 1e-30))
            p = r + beta * (p - omega * v)
            p_hat = M_apply(p)
            v = mv(p_hat)
            denom = torch.sum(r_hat * v) + 1e-30
            alpha = rho_new / denom
            s = r - alpha * v
            if torch.norm(s) < tol:
                x = x + alpha * p_hat
                break
            s_hat = M_apply(s)
            t = mv(s_hat)
            tt = torch.sum(t * t) + 1e-30
            omega = torch.sum(t * s) / tt
            x = x + alpha * p_hat + omega * s_hat
            r = s - omega * t
            if torch.norm(r) < tol or omega.abs() < 1e-30:
                break
            rho = rho_new
        return x

    def cg_solve(self, mv, rhs, x0=None, max_iter=30, tol=1e-4):
        x = torch.zeros_like(rhs) if x0 is None else x0.clone()
        r = rhs - mv(x)
        p = r.clone()
        rs_old = torch.sum(r * r)
        for _ in range(max_iter):
            Ap = mv(p)
            alpha = rs_old / (torch.sum(p * Ap) + 1e-12)
            x = x + alpha * p
            r = r - alpha * Ap
            rs_new = torch.sum(r * r)
            if torch.sqrt(rs_new) <= tol:
                break
            beta = rs_new / (rs_old + 1e-12)
            p = r + beta * p
            rs_old = rs_new
        return x

    def forward(self, u_now: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor,
                dx_col=0, dz_col=1, slope_col=None):
        N = u_now.size(0)
        self._ensure_built(N, edge_index, edge_attr, dx_col=dx_col, dz_col=dz_col, slope_col=slope_col)

        dt_eff = torch.clamp(self.dt, self.dt_min, self.dt_max)
        S = slope_term_nodes(edge_index, self.slope_e, N, like=u_now)
        b = u_now - dt_eff * self.g_hat * S

        if self.solver == "fs":
            return self._solve_fs(u_now, b, dt_eff)
        elif self.solver == "jacobi":
            return self._solve_jacobi(u_now, u_now, b, dt_eff)
        elif self.solver == "bicgstab":
            mv = lambda v: self.A_mv(v, u_now, edge_index, dt_eff)
            Minv = self._diag_prec_A(u_now, dt_eff)
            return self._bicgstab(mv, b, x0=u_now, max_iter=self.krylov_max_iter, tol=self.krylov_tol, Minv=Minv)
        elif self.solver == "necg":
            ATb = self.AT_mv(b, u_now, edge_index, dt_eff)
            lam = float(self.lambda_reg)

            def Mv(v):
                Av = self.A_mv(v, u_now, edge_index, dt_eff)
                ATAv = self.AT_mv(Av, u_now, edge_index, dt_eff)
                if lam > 0:
                    ATAv = ATAv + lam * self.laplacian(v, edge_index)
                return ATAv

            return self.cg_solve(Mv, ATb, x0=u_now, max_iter=self.cg_max_iter, tol=self.cg_tol)
        else:
            raise ValueError(f"Unknown solver: {self.solver}")


# ==========================
# Ghost (lightweight): top-k energy mask + optional loss weighting
# ==========================
class GhostController(nn.Module):
    """
    极简 ghost：每个 forward 的 decoder rollout 里累积能量 E(i)=Σ r^2(i) (可选*diagA)
    然后用 topk_ratio 触发 active mask；可做 EMA 平滑。
    """
    def __init__(self, topk_ratio=0.12, ema_alpha=0.25, temporal=True, tau_off_frac=0.75):
        super().__init__()
        self.topk_ratio = float(topk_ratio)
        self.ema_alpha = float(ema_alpha)
        self.temporal = bool(temporal)
        self.tau_off_frac = float(tau_off_frac)

        self._ema = None
        self._prev_mask = None

    def reset_if_needed(self, N, device):
        if self._ema is None or self._ema.size(0) != N:
            self._ema = torch.zeros((N, 1), device=device)
        if self._prev_mask is None or self._prev_mask.size(0) != N:
            self._prev_mask = torch.zeros((N, 1), device=device, dtype=torch.bool)

    @torch.no_grad()
    def update(self, strength: torch.Tensor):
        # strength: [N,1]
        N = strength.size(0)
        device = strength.device
        self.reset_if_needed(N, device)

        if self.temporal:
            self._ema = (1.0 - self.ema_alpha) * self._ema + self.ema_alpha * strength
            ref = self._ema
        else:
            ref = strength

        k = max(1, int(math.ceil(self.topk_ratio * N)))
        vals, _ = torch.topk(ref.detach().flatten(), k)
        tau_on = vals.min()
        tau_off = tau_on * self.tau_off_frac

        prev = self._prev_mask
        on_now = (ref >= tau_on)
        keep_on = (ref >= tau_off) & prev
        mask = (on_now | keep_on).detach()
        self._prev_mask = mask
        return mask

    @property
    def mask(self):
        return self._prev_mask


def weighted_mse(y_pred: torch.Tensor, y_true: torch.Tensor,
                 node_weight: Optional[torch.Tensor] = None,
                 label_mask: Optional[torch.Tensor] = None):
    # y_pred/y_true: [N,T]
    se = (y_pred - y_true) ** 2
    if label_mask is None:
        label_mask = torch.isfinite(y_true).float()
    se = torch.where(label_mask > 0, se, torch.zeros_like(se))
    if node_weight is None:
        denom = label_mask.sum().clamp_min(1.0)
        return se.sum() / denom
    w = node_weight.view(-1, 1)
    denom = (w * label_mask).sum().clamp_min(1.0)
    return (w * se).sum() / denom


# ==========================
# Train / Eval / Plot
# ==========================
def train_epoch(model, loader, optimizer, device,
                tf_ratio=0.0, boundary_tf_ratio=0.0,
                ghost_weight_mode="none", ghost_beta=0.0,
                max_grad_norm: Optional[float] = 1.0):
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        y_true = batch.y.squeeze(-1)  # [N,T]
        y_pred = model(batch, y_target=y_true,
                       teacher_forcing_ratio=tf_ratio,
                       boundary_tf_ratio=boundary_tf_ratio)

        if ghost_weight_mode == "none" or ghost_beta <= 0:
            loss = weighted_mse(y_pred, y_true)
        else:
            if hasattr(model, "ghost") and model.ghost is not None and model.ghost.mask is not None:
                m = model.ghost.mask.float().squeeze(-1)
            else:
                m = torch.zeros((y_true.size(0),), device=device)
            if ghost_weight_mode == "active":
                w = 1.0 + float(ghost_beta) * m
            else:
                w = 1.0 + float(ghost_beta) * m  # 这里先同 active（极简）
            loss = weighted_mse(y_pred, y_true, node_weight=w)

        loss.backward()
        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(1, n_batches)


@torch.no_grad()
def evaluate_epoch(model, loader, device):
    model.eval()
    total_mse = 0.0
    total_mae = 0.0
    num_elems = 0
    for batch in loader:
        batch = batch.to(device)
        y_true = batch.y.squeeze(-1)
        y_pred = model(batch, y_target=None, teacher_forcing_ratio=0.0, boundary_tf_ratio=0.0)
        mse, mae = metric_batch(y_pred, y_true)
        cnt = y_pred.numel()
        total_mse += mse * cnt
        total_mae += mae * cnt
        num_elems += cnt
    avg_mse = total_mse / max(1, num_elems)
    avg_mae = total_mae / max(1, num_elems)
    avg_rmse = float(np.sqrt(avg_mse))
    return avg_mse, avg_mae, avg_rmse


def plot_learning_curves(save_dir: str, train_losses: List[float], val_rmses: List[float]):
    import matplotlib.pyplot as plt
    make_dir(save_dir)
    xs = np.arange(1, len(train_losses) + 1)

    plt.figure()
    plt.plot(xs, train_losses)
    plt.xlabel("Epoch")
    plt.ylabel("Train Loss (MSE)")
    plt.title("Training Curve")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "train_curve.png"), dpi=200)
    plt.close()

    plt.figure()
    plt.plot(xs, val_rmses)
    plt.xlabel("Epoch")
    plt.ylabel("Val RMSE")
    plt.title("Validation RMSE")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "val_rmse.png"), dpi=200)
    plt.close()


@torch.no_grad()
def plot_sample_rollout(save_dir: str, model, sample, device, title_prefix="sample"):
    import matplotlib.pyplot as plt
    make_dir(save_dir)
    model.eval()
    sample = sample.to(device)
    y_true = sample.y.squeeze(-1)         # [N,T]
    y_pred = model(sample, y_target=None, teacher_forcing_ratio=0.0, boundary_tf_ratio=0.0)  # [N,T]

    N, T = y_true.shape
    # 随机挑 6 个节点画曲线
    k = min(6, N)
    idx = torch.randperm(N, device=device)[:k].detach().cpu().numpy().tolist()

    plt.figure()
    for i in idx:
        plt.plot(np.arange(T), y_true[i].detach().cpu().numpy(), linestyle='-')
        plt.plot(np.arange(T), y_pred[i].detach().cpu().numpy(), linestyle='--')
    plt.xlabel("Horizon t")
    plt.ylabel("y")
    plt.title(f"{title_prefix}: truth(solid) vs pred(dashed) for {k} nodes")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{title_prefix}_rollout.png"), dpi=200)
    plt.close()

    # 再画一个：每个节点的 RMSE 分布（柱状/直方）
    err = (y_pred - y_true).detach().cpu().numpy()
    rmse_nodes = np.sqrt(np.mean(err ** 2, axis=1))

    plt.figure()
    plt.hist(rmse_nodes, bins=40)
    plt.xlabel("Node RMSE")
    plt.ylabel("Count")
    plt.title(f"{title_prefix}: node RMSE histogram")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{title_prefix}_node_rmse_hist.png"), dpi=200)
    plt.close()


def run_experiment(hparams: Dict[str, Any],
                   build_model_fn: Callable[[Any, Dict[str, Any]], nn.Module]):
    """
    统一入口：加载数据->split->train->eval->save best->画图
    build_model_fn(sample, hparams) 返回 model
    """
    ensure_reproducibility(hparams["seed"])
    device = torch.device("cuda" if (torch.cuda.is_available() and hparams.get("use_cuda", True)) else "cpu")
    print(f"[Device] {device}")

    dcfg = hparams["data"]
    dataset = LamaHDataset(
        root_dir=dcfg["path"],
        years=dcfg["years_total"],
        root_gauge_id=dcfg["root_gauge_id"],
        rewire_graph=dcfg["rewire_graph"],
        input_window_size=dcfg["input_seq_length"],
        total_target_window=dcfg["target_seq_length"],
        stride_length=dcfg["stride_length"],
        normalized=dcfg["normalized"],
    )
    if len(dataset) == 0:
        raise RuntimeError("Dataset is empty.")

    sample = dataset[0]
    print(f"[Data] samples={len(dataset)}, nodes={sample.num_nodes}, "
          f"F={sample.x.shape[-1]}, Tin={sample.x.shape[1]}, Tout={sample.y.shape[1]}")

    size = len(dataset)
    test_size = int(hparams["training"]["test_split"] * size)
    val_size = int(hparams["training"]["val_split"] * size)
    train_size = size - val_size - test_size
    if min(train_size, val_size, test_size) <= 0:
        raise RuntimeError("Bad split sizes.")

    g = torch.Generator().manual_seed(hparams["seed"])
    train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size], generator=g)

    tcfg = hparams["training"]
    train_loader = PyGDataLoader(train_set, batch_size=tcfg["batch_size"], shuffle=True, drop_last=True, num_workers=0)
    val_loader = PyGDataLoader(val_set, batch_size=tcfg["batch_size"], shuffle=False, drop_last=False, num_workers=0)

    model = build_model_fn(sample, hparams).to(device)
    print(model)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Params] trainable={total_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=tcfg["learning_rate"])

    save_path = hparams["save_path"]
    make_dir(save_path)
    plot_dir = os.path.join(save_path, "plots")
    make_dir(plot_dir)

    best_val = float("inf")
    best_state = None
    train_losses, val_rmses = [], []
    patience = int(tcfg.get("early_stopping_patience", 80))
    no_improve = 0

    ghost_mode = tcfg.get("ghost_weight_mode", "none").lower()
    ghost_beta = float(tcfg.get("ghost_weight_beta", 0.0))

    for epoch in range(1, tcfg["epochs"] + 1):
        tf_ratio = float(tcfg.get("teacher_forcing_ratio", 0.0))
        btf_ratio = float(tcfg.get("boundary_tf_ratio", 0.0))

        tr = train_epoch(model, train_loader, optimizer, device,
                         tf_ratio=tf_ratio, boundary_tf_ratio=btf_ratio,
                         ghost_weight_mode=ghost_mode, ghost_beta=ghost_beta,
                         max_grad_norm=tcfg.get("max_grad_norm", 1.0))
        val_mse, val_mae, val_rmse = evaluate_epoch(model, val_loader, device)

        train_losses.append(tr)
        val_rmses.append(val_rmse)

        print(f"[Epoch {epoch:03d}] train={tr:.6f} | val_rmse={val_rmse:.6f} (mse={val_mse:.6f}, mae={val_mae:.6f})")

        if val_rmse < best_val:
            best_val = val_rmse
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"[EarlyStop] epoch={epoch}, best_val_rmse={best_val:.6f}")
                break

    if best_state is not None:
        torch.save(best_state, os.path.join(save_path, "best.pt"))
        model.load_state_dict(best_state)

    # plots
    plot_learning_curves(plot_dir, train_losses, val_rmses)
    plot_sample_rollout(plot_dir, model, dataset[0], device, title_prefix="sample0")

    return {
        "train_losses": train_losses,
        "val_rmses": val_rmses,
        "best_val_rmse": best_val,
        "save_path": save_path,
    }
