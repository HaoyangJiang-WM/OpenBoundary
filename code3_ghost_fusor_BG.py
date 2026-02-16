# code3_ghost_fusor_BG.py
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

from stgnn_common import (
    run_experiment, GNNLayer, RNNBlock, MLPPredictor,
    BoundaryFuser, GhostController,
    boundary_and_downstream_indices, first_edge_value_per_src,
    PhysicsExplicitGStep
)

class STGNN_AR_GhostFusor_BG(nn.Module):
    def __init__(self, node_features, hidden_dim, Tin, Tout,
                 rnn_type="gru", dropout=0.1, num_gnn_layers=3,
                 alpha_boundary=0.5, boundary_lag=0,
                 distance_col=0, dz_col=1, slope_col=None,
                 use_B=True, use_G=True,
                 bg_cycles=2, boundary_relax=0.7,
                 g_dt=0.2, g_mode="symmetric"):
        super().__init__()
        self.Tin, self.Tout = Tin, Tout
        self.alpha_boundary = float(alpha_boundary)
        self.boundary_lag = int(boundary_lag)
        self.distance_col = int(distance_col)
        self.dz_col = int(dz_col) if dz_col is not None else 1
        self.slope_col = slope_col

        self.use_B = bool(use_B)
        self.use_G = bool(use_G)
        self.bg_cycles = int(bg_cycles)
        self.boundary_relax = float(boundary_relax)

        self.gnns = nn.ModuleList()
        self.gnns.append(GNNLayer(node_features, hidden_dim, dropout))
        for _ in range(num_gnn_layers - 1):
            self.gnns.append(GNNLayer(hidden_dim, hidden_dim, dropout))

        self.rnn = RNNBlock(hidden_dim, hidden_dim, rnn_type=rnn_type)
        self.drop = nn.Dropout(dropout)
        self.head = MLPPredictor(hidden_dim, hidden_dim, 1, num_layers=3, dropout=dropout)

        self.fuser_enc = BoundaryFuser(node_features, hidden=max(16, 2 * node_features), dropout=dropout)
        self.fuser_dec = BoundaryFuser(1, hidden=32, dropout=dropout)

        self.ghost = GhostController(topk_ratio=0.12, ema_alpha=0.25, temporal=True, tau_off_frac=0.75)

        # Explicit G
        self.physics_g = PhysicsExplicitGStep(dt_init=g_dt, mode=g_mode)

    def _init_h(self, N, device):
        if isinstance(self.rnn.rnn_cell, nn.LSTMCell):
            h = torch.zeros(N, self.head.layers[0].in_features, device=device)
            c = torch.zeros(N, self.head.layers[0].in_features, device=device)
            return (h, c)
        return torch.zeros(N, self.head.layers[0].in_features, device=device)

    def _run_gnn(self, x_t, edge_index, edge_attr):
        ew = None
        if edge_attr is not None and edge_attr.size(1) > self.distance_col:
            dist = edge_attr[:, self.distance_col].clamp_min(1e-6)
            ew = 1.0 / dist
        h = x_t
        for g in self.gnns:
            h = g(h, edge_index, ew)
        return h

    def _apply_fuse(self, node_feat, bnd, down_feat, inv_dx_bnd, encoder_stage: bool):
        if bnd.numel() == 0:
            return node_feat
        self_feat = node_feat[bnd]
        diff_norm = (self_feat - down_feat) * inv_dx_bnd
        fuser = self.fuser_enc if encoder_stage else self.fuser_dec
        delta = fuser(self_feat, down_feat, diff_norm)
        out = node_feat.clone()
        out[bnd] = self_feat + self.alpha_boundary * delta
        return out

    @staticmethod
    def _robin_relax(ub_cur, ub_anchor, relax=0.7):
        return (1.0 - relax) * ub_cur + relax * ub_anchor

    def forward(self, data, y_target=None, teacher_forcing_ratio=0.0, boundary_tf_ratio=0.0):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        device = x.device
        N = x.size(0)

        bnd, down = boundary_and_downstream_indices(edge_index, N)
        dist = edge_attr[:, self.distance_col].clamp_min(1e-6)
        dx_first = first_edge_value_per_src(edge_index, dist, N)
        dx_bnd = dx_first[bnd].clamp_min(1e-6).unsqueeze(-1) if bnd.numel() > 0 else None
        inv_dx_bnd = (1.0 / dx_bnd) if dx_bnd is not None else None

        h = self._init_h(N, device)
        hist = []

        # encoder
        for t in range(self.Tin):
            x_t = x[:, t, :]
            if bnd.numel() > 0:
                x_t = self._apply_fuse(x_t, bnd, x_t[down], inv_dx_bnd, encoder_stage=True)
            g = self._run_gnn(x_t, edge_index, edge_attr)
            h = self.rnn(g, h)
            if isinstance(h, tuple):
                h = (self.drop(h[0]), h[1])
            else:
                h = self.drop(h)
            hist.append(x_t.detach())

        # decoder AR
        cur = hist[-1][:, 0:1] if hist[-1].size(1) > 1 else hist[-1]
        preds = []
        dec_hist = [cur.detach()]

        ghost_strength = torch.zeros((N, 1), device=device)

        for t in range(self.Tout):
            g_in = cur.expand(-1, x.size(-1)) if cur.size(1) == 1 else cur
            g = self._run_gnn(g_in, edge_index, edge_attr)
            h = self.rnn(g, h)
            h_use = self.drop(h[0]) if isinstance(h, tuple) else self.drop(h)
            y = self.head(h_use)  # [N,1]
            preds.append(y)

            use_tf = (self.training and y_target is not None and (torch.rand(1).item() < teacher_forcing_ratio))
            nxt = (y_target[:, t].unsqueeze(-1) if use_tf else y).clone()

            ub_anchor = None
            if bnd.numel() > 0:
                d_pred = (dec_hist[-self.boundary_lag][down] if (self.boundary_lag > 0 and len(dec_hist) > self.boundary_lag)
                          else y[down]).detach()
                if self.training and y_target is not None and boundary_tf_ratio > 0:
                    d_gt = y_target[down, t].unsqueeze(-1)
                    m = (torch.rand(len(bnd), device=device) < boundary_tf_ratio).float().unsqueeze(-1)
                    d_for = m * d_gt + (1.0 - m) * d_pred
                else:
                    d_for = d_pred
                nxt = self._apply_fuse(nxt, bnd, d_for, inv_dx_bnd, encoder_stage=False)
                ub_anchor = nxt[bnd].detach()

            # ===== BG cycles =====
            for _ in range(max(1, self.bg_cycles)):
                if self.use_B and bnd.numel() > 0 and ub_anchor is not None:
                    ub_new = self._robin_relax(nxt[bnd], ub_anchor, relax=self.boundary_relax)
                    nxt = nxt.clone()
                    nxt[bnd] = ub_new

                if self.use_G:
                    nxt = self.physics_g(nxt, edge_index, edge_attr,
                                         dx_col=self.distance_col, dz_col=self.dz_col, slope_col=self.slope_col)

            ghost_strength += (nxt * nxt).detach()

            cur = nxt
            dec_hist.append(cur.detach())

        if self.training:
            self.ghost.update(ghost_strength)

        y_pred = torch.cat(preds, dim=1)
        return y_pred


def build_model(sample, hparams):
    m = hparams["model"]
    return STGNN_AR_GhostFusor_BG(
        node_features=sample.x.shape[-1],
        hidden_dim=m["hidden_dim"],
        Tin=hparams["data"]["input_seq_length"],
        Tout=hparams["data"]["target_seq_length"],
        rnn_type=m.get("rnn_type", "gru"),
        dropout=m.get("dropout", 0.1),
        num_gnn_layers=m.get("num_gnn_layers", 3),
        distance_col=m.get("distance_col", 0),
        dz_col=m.get("dz_col", 1),
        slope_col=m.get("slope_col", None),
        alpha_boundary=m.get("alpha_boundary", 0.5),
        boundary_lag=m.get("boundary_lag", 0),
        use_B=m.get("use_B", True),
        use_G=m.get("use_G", True),
        bg_cycles=m.get("bg_cycles", 2),
        boundary_relax=m.get("boundary_relax", 0.7),
        g_dt=m.get("g_dt", 0.2),
        g_mode="symmetric",
    )


if __name__ == "__main__":
    hparams = {
        "seed": 42,
        "use_cuda": True,
        "save_path": "runs_code3_ghost_fusor_BG",
        "data": {
            "path": "./LamaH-CE",
            "years_total": list(range(2000, 2005)),
            "root_gauge_id": 399,
            "rewire_graph": True,
            "input_seq_length": 24,
            "target_seq_length": 6,
            "stride_length": 1,
            "normalized": True,
        },
        "model": {
            "hidden_dim": 64,
            "rnn_type": "gru",
            "dropout": 0.1,
            "num_gnn_layers": 3,
            "distance_col": 0,
            "dz_col": 1,
            "slope_col": None,

            "alpha_boundary": 0.5,
            "boundary_lag": 0,

            "use_B": True,
            "use_G": True,
            "bg_cycles": 2,
            "boundary_relax": 0.7,
            "g_dt": 0.2,
        },
        "training": {
            "batch_size": 32,
            "epochs": 300,
            "learning_rate": 1e-3,
            "val_split": 0.15,
            "test_split": 0.15,
            "early_stopping_patience": 80,
            "teacher_forcing_ratio": 0.2,
            "boundary_tf_ratio": 0.8,
            "max_grad_norm": 1.0,
            "ghost_weight_mode": "active",
            "ghost_weight_beta": 0.1,
        },
    }
    run_experiment(hparams, build_model)
