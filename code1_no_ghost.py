# code1_no_ghost.py
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

from stgnn_common import (
    run_experiment, GNNLayer, RNNBlock, MLPPredictor
)

class STGNN_AR_NoGhost(nn.Module):
    def __init__(self, node_features, hidden_dim, Tin, Tout,
                 rnn_type="gru", dropout=0.1, num_gnn_layers=3,
                 distance_col=0):
        super().__init__()
        self.Tin, self.Tout = Tin, Tout
        self.distance_col = int(distance_col)

        self.gnns = nn.ModuleList()
        self.gnns.append(GNNLayer(node_features, hidden_dim, dropout))
        for _ in range(num_gnn_layers - 1):
            self.gnns.append(GNNLayer(hidden_dim, hidden_dim, dropout))

        self.rnn = RNNBlock(hidden_dim, hidden_dim, rnn_type=rnn_type)
        self.drop = nn.Dropout(dropout)
        self.head = MLPPredictor(hidden_dim, hidden_dim, 1, num_layers=3, dropout=dropout)

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

    def forward(self, data, y_target=None, teacher_forcing_ratio=0.0, boundary_tf_ratio=0.0):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        device = x.device
        N = x.size(0)

        h = self._init_h(N, device)
        # encoder
        last_x = None
        for t in range(self.Tin):
            x_t = x[:, t, :]
            last_x = x_t
            g = self._run_gnn(x_t, edge_index, edge_attr)
            h = self.rnn(g, h)
            if isinstance(h, tuple):
                h = (self.drop(h[0]), h[1])
            else:
                h = self.drop(h)

        # decoder AR
        cur = last_x[:, 0:1] if last_x.size(1) > 1 else last_x  # 防呆
        preds = []
        for t in range(self.Tout):
            g = self._run_gnn(cur.expand(-1, x.size(-1)) if cur.size(1)==1 else cur, edge_index, edge_attr)
            h = self.rnn(g, h)
            h_use = self.drop(h[0]) if isinstance(h, tuple) else self.drop(h)
            y = self.head(h_use)  # [N,1]
            preds.append(y)

            use_tf = (self.training and y_target is not None and (torch.rand(1).item() < teacher_forcing_ratio))
            cur = (y_target[:, t].unsqueeze(-1) if use_tf else y)

        y_pred = torch.cat(preds, dim=1)  # [N,Tout]
        return y_pred


def build_model(sample, hparams):
    m = hparams["model"]
    return STGNN_AR_NoGhost(
        node_features=sample.x.shape[-1],
        hidden_dim=m["hidden_dim"],
        Tin=hparams["data"]["input_seq_length"],
        Tout=hparams["data"]["target_seq_length"],
        rnn_type=m.get("rnn_type", "gru"),
        dropout=m.get("dropout", 0.1),
        num_gnn_layers=m.get("num_gnn_layers", 3),
        distance_col=m.get("distance_col", 0),
    )


if __name__ == "__main__":
    hparams = {
        "seed": 42,
        "use_cuda": True,
        "save_path": "runs_code1_no_ghost",
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
        },
        "training": {
            "batch_size": 32,
            "epochs": 200,
            "learning_rate": 1e-3,
            "val_split": 0.15,
            "test_split": 0.15,
            "early_stopping_patience": 60,
            "teacher_forcing_ratio": 0.2,
            "boundary_tf_ratio": 0.0,
            "max_grad_norm": 1.0,
            "ghost_weight_mode": "none",
            "ghost_weight_beta": 0.0,
        },
    }
    run_experiment(hparams, build_model)
