# code4_ghost_fusor_BG_upwind.py
# -*- coding: utf-8 -*-

from code3_ghost_fusor_BG import STGNN_AR_GhostFusor_BG, build_model as _build_base
from stgnn_common import run_experiment

def build_model(sample, hparams):
    # 复用 code3 的 build_model，但强制 g_mode=upwind
    m = hparams["model"]
    model = _build_base(sample, hparams)
    # 直接改 physics_g 的模式（或者你也可以在构造时传 g_mode）
    model.physics_g.mode = "upwind"
    return model

if __name__ == "__main__":
    hparams = {
        "seed": 42,
        "use_cuda": True,
        "save_path": "runs_code4_ghost_fusor_BG_upwind",
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
