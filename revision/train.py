"""
Model factory, predeclared hyperparameter grids and the training loop of the
revised protocol.  The loop is src/deep_models.train_torch with one change:
the validation loss that drives early stopping can be restricted to observed
target hours (mask).  With an all-true mask it equals the original loop.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

import common as C
import deep_models as DM

torch.backends.cudnn.benchmark = False

# Predeclared grids (fixed before any revised run).  Each grid contains the
# configuration used in the submission.
GRIDS = {
    "MLP": [dict(hidden=h, dropout=p) for h in (64, 128) for p in (0.15, 0.3)],
    "LSTM": [dict(hidden=h, layers=k) for h in (24, 48) for k in (1, 2)],
    "Transformer": [dict(d_model=d, heads=a, layers=k, ff=f, dropout=p)
                    for (d, a, k, f) in ((16, 1, 1, 24), (32, 4, 2, 64)) for p in (0.15, 0.3)],
    "CLEAR-Former": [dict(d_model=d, heads=a, layers=k, ff=f, dropout=p)
                     for (d, a, k, f) in ((16, 1, 1, 24), (40, 4, 2, 80)) for p in (0.15, 0.3)],
    "SVR": [dict(C=c) for c in (1.0, 10.0, 100.0)],
    "RandomForest": [dict(max_depth=d) for d in (6, 12, None)],
    "XGBoost": [dict(max_depth=d) for d in (2, 4, 6)],
}
SUBMITTED = {
    "MLP": dict(hidden=128, dropout=0.15), "LSTM": dict(hidden=48, layers=1),
    "Transformer": dict(d_model=32, heads=4, layers=2, ff=64, dropout=0.15),
    "CLEAR-Former": dict(d_model=16, heads=1, layers=1, ff=24, dropout=0.3),
}
NEURAL = ("MLP", "LSTM", "Transformer", "CLEAR-Former")
N_FEAT = 13
N_STATIC = 41   # 24 load lags, 5 window statistics, 7 covariates and 5 calendar values at the origin


def build_neural(kind, cfg, H, seed, ablate=None):
    torch.manual_seed(seed)
    if kind == "MLP":
        return DM.MLPForecaster(N_FEAT, C.L, H, hidden=cfg["hidden"], dropout=cfg["dropout"])
    if kind == "LSTM":
        return DM.LSTMForecaster(N_FEAT, H, hidden=cfg["hidden"], layers=cfg["layers"])
    if kind == "Transformer":
        return DM.TransformerTS(N_FEAT, C.L, H, d_model=cfg["d_model"], heads=cfg["heads"],
                                layers=cfg["layers"], ff=cfg["ff"], dropout=cfg["dropout"])
    if kind == "CLEAR-Former":
        flags = dict(use_clim=True, use_decomp=True, use_featmlp=True, use_attn=True)
        flags.update(ablate or {})
        return DM.ApartElectricFormer(N_FEAT, C.L, H, target_idx=C.TARGET_IDX, cov_slice=C.COV_SLICE,
                                      n_static=N_STATIC, d_model=cfg["d_model"], heads=cfg["heads"],
                                      layers=cfg["layers"], ff=cfg["ff"], dropout=cfg["dropout"], **flags)
    raise ValueError(kind)


def uses_static(kind, ablate=None):
    return kind == "CLEAR-Former" and (ablate or {}).get("use_featmlp", True)


def _t(a, device, dtype=torch.float32):
    return None if a is None else torch.tensor(a, dtype=dtype, device=device)


def train_masked(model, Xtr, ytr, hidtr, dowtr, Xva, yva, hidva, dowva, va_mask, device,
                 epochs=200, patience=25, lr=1e-3, wd=1e-4, batch=32, seed=0, sftr=None, sfva=None):
    """src/deep_models.train_torch with a masked validation loss."""
    torch.manual_seed(seed); np.random.seed(seed)
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=8)
    loss_fn = nn.SmoothL1Loss()
    loss_el = nn.SmoothL1Loss(reduction="none")
    Xtr, ytr, Xva, yva = _t(Xtr, device), _t(ytr, device), _t(Xva, device), _t(yva, device)
    hidtr, dowtr = _t(hidtr, device, torch.long), _t(dowtr, device, torch.long)
    hidva, dowva = _t(hidva, device, torch.long), _t(dowva, device, torch.long)
    sftr, sfva = _t(sftr, device), _t(sfva, device)
    mask = _t(va_mask.astype(np.float32), device)
    n_mask = float(mask.sum().item())
    n = Xtr.size(0)
    best, best_state, wait, best_ep, ep = float("inf"), None, 0, -1, -1
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            pred = model(Xtr[idx], hidtr[idx], dowtr[idx], None if sftr is None else sftr[idx])
            loss = loss_fn(pred, ytr[idx])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            vpred = model(Xva, hidva, dowva, sfva)
            vloss = float((loss_el(vpred, yva) * mask).sum().item() / n_mask)
        sched.step(vloss)
        if vloss < best - 1e-6:
            best, wait, best_ep = vloss, 0, ep
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {"best_epoch": best_ep + 1, "epochs_run": ep + 1, "best_val_loss": best}


def n_params(model):
    return int(sum(p.numel() for p in model.parameters()))
