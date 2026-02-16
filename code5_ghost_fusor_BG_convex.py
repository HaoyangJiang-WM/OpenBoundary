# code4_ghost_fusor_BG_convex.py
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

from stgnn_common import (
    run_experiment, PhysicsLSGStep
)
from code3_ghost_fusor_BG import STGNN_AR_GhostFusor_BG

class STGNN_AR_GhostFusor_BG_Convex(STGNN_AR_GhostFusor_BG):
    def __init__(self, *args,
                 use_convex=True,
                 convex_lambda=0.0,
                 convex_dt=0.3,
                 convex_solver="necg",
                 convex_cg_max_iter=30,
                 convex_cg_tol=1e-4,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.use_convex = bool(use_convex)
        self.convex = PhysicsLSGStep(
            lambda_reg=convex_lambda,
            dt_init=convex_dt,
            solver=convex_solver,
            cg_max_iter=convex_cg_max_iter,
            cg_tol=convex_cg_tol,
        )

    def forward(self, data, y_target=None, teacher_forcing_ratio=0.0, boundary_tf_ratio=0.0):
        # 直接复用父类逻辑，但在 BG cycles 后追加 convex refiner
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        device = x.device
        N = x.size(0)

        # --- 复制父类 forward 的主体（最小改动：在 BG cycles 后加 convex） ---
        from stgnn_common import boundary_and_downstream_indices, first_edge_value_per_src

        bnd, down = boundary_and_downstream_indices(edge_index, N)
        dist = edge_attr[:, self.distance_col].clamp_min(1e-6)
        dx_first = first_edge_value_per_src(edge_index, dist, N)
        dx_bnd = dx_first[bnd].clamp_min(1e-6).unsqueeze(-1) if bnd.numel() > 0 else None
        inv_dx_bnd = (1.0 / dx_bnd) if dx_bnd is not None else None

        h = self._init_h(N, device)
        hist = []

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

        cur = hist[-1][:, 0:1] if hist[-1].size(1) > 1 else hist[-1]
        preds = []
        dec_hist = [cur.detach()]
        ghost_strength = torch.zeros((N, 1), device=device)

        for t in range(self.Tout):
            g_in = cur.expand(-1, x.size(-1)) if cur.size(1) == 1 else cur
            g = self._run_gnn(g_in, edge_index, edge_attr)
            h = self.rnn(g, h)
            h_use = self.drop(h[0]) if isinstance(h, tuple) else self.drop(h)
            y = self.head(h_use)
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

            # BG cycles (父类)
            for _ in range(max(1, self.bg_cycles)):
                if self.use_B and bnd.numel() > 0 and ub_anchor is not None:
                    ub_new = self._robin_relax(nxt[bnd], ub_anchor, relax=self.boundary_relax)
                    nxt = nxt.clone()
                    nxt[bnd] = ub_new

                if self.use_G:
                    nxt = self.physics_g(nxt, edge_index, edge_attr,
                                         dx_col=self.distance_col, dz_col=self.dz_col, slope_col=self.slope_col)

            # === Convexified refiner (新增) ===
            if self.use_convex:
                nxt = self.convex(nxt, edge_index, edge_attr,
                                  dx_col=self.distance_col, dz_col=self.dz_col, slope_col=self.slope_col)

            ghost_strength += (nxt * nxt).detach()
            cur = nxt
            dec_hist.append(cur.detach())

        if self.training:
            self.ghost.update(ghost_strength)

        return torch.cat(preds, dim=1)


def build_model(sample, hparams):
    m = hparams["model"]
    return STGNN_AR_GhostFusor_BG_Convex(
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

        # convex
        use_convex=m.get("use_convex", True),
        convex_lambda=m.get("convex_lambda", 0.0),
        convex_dt=m.get("convex_dt", 0.3),
        convex_solver=m.get("convex_solver", "necg"),  # "fs" / "jacobi" / "bicgstab" / "necg"
        convex_cg_max_iter=m.get("convex_cg_max_iter", 30),
        convex_cg_tol=m.get("convex_cg_tol", 1e-4),
    )


if __name__ == "__main__":
    hparams = {
        "seed": 42,
        "use_cuda": True,
        "save_path": "runs_code5_BG_convex",
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

            "use_convex": True,
            "convex_lambda": 0.0,
            "convex_dt": 0.3,
            "convex_solver": "necg",
            "convex_cg_max_iter": 30,
            "convex_cg_tol": 1e-4,
        },
        "training": {
            "batch_size": 32,
            "epochs": 350,
            "learning_rate": 1e-3,
            "val_split": 0.15,
            "test_split": 0.15,
            "early_stopping_patience": 90,
            "teacher_forcing_ratio": 0.2,
            "boundary_tf_ratio": 0.8,
            "max_grad_norm": 1.0,
            "ghost_weight_mode": "active",
            "ghost_weight_beta": 0.1,
        },
    }
    run_experiment(hparams, build_model)
