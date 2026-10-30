"""
Deep sequence models in PyTorch for apartment STLF:
  * MLP           - flatten-window multilayer perceptron
  * TransformerTS - vanilla encoder-only Transformer over per-hour tokens
  * ApartElectricFormer - the proposed model:
        (i)   series decomposition of the load channel (trend / seasonal)
        (ii)  PatchTST-style patch embedding of the load window
        (iii) periodicity-aware positional encoding
              (learnable hour-of-day and day-of-week embeddings)
        (iv)  gated fusion of electrical power-quality covariates (V, I, PF)
        (v)   encoder-only Transformer + direct multi-horizon linear head

All models are intentionally small (a few k..~50k params) because the data
set is short (~450 hourly points).  Same infra => the vanilla Transformer is a
clean ablation baseline for the proposed model.
"""
from __future__ import annotations
import math
import numpy as np
import torch
import torch.nn as nn


# --------------------------------------------------------------------------- #
# LSTM
# --------------------------------------------------------------------------- #
class LSTMForecaster(nn.Module):
    def __init__(self, n_feat, out_dim, hidden=48, layers=1, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(n_feat, hidden, num_layers=layers,
                            batch_first=True, dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, out_dim))

    def forward(self, x, hid=None, dow=None, sf=None):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])


# --------------------------------------------------------------------------- #
# MLP
# --------------------------------------------------------------------------- #
class MLPForecaster(nn.Module):
    def __init__(self, n_feat, seq_len, out_dim, hidden=128, dropout=0.15):
        super().__init__()
        d = n_feat * seq_len
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(d, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden // 2, out_dim),
        )

    def forward(self, x, hid=None, dow=None, sf=None):
        return self.net(x)


# --------------------------------------------------------------------------- #
# Vanilla Transformer (encoder-only) over per-hour tokens
# --------------------------------------------------------------------------- #
class _SinPos(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class TransformerTS(nn.Module):
    def __init__(self, n_feat, seq_len, out_dim, d_model=32, heads=4,
                 layers=2, ff=64, dropout=0.15):
        super().__init__()
        self.embed = nn.Linear(n_feat, d_model)
        self.pos = _SinPos(d_model, max_len=seq_len + 8)
        enc = nn.TransformerEncoderLayer(d_model, heads, ff, dropout,
                                         batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(enc, layers)
        self.head = nn.Sequential(nn.Flatten(),
                                  nn.Linear(d_model * seq_len, out_dim))

    def forward(self, x, hid=None, dow=None, sf=None):
        z = self.pos(self.embed(x))
        z = self.encoder(z)
        return self.head(z)


# --------------------------------------------------------------------------- #
# ApartElectric-Former (proposed)
# --------------------------------------------------------------------------- #
class _SeriesDecomp(nn.Module):
    """Moving-average trend/seasonal decomposition (Autoformer/DLinear style)."""
    def __init__(self, kernel=25):
        super().__init__()
        self.kernel = kernel
        self.avg = nn.AvgPool1d(kernel, stride=1, padding=0)

    def forward(self, x):                       # x: (B, L) single channel
        pad = (self.kernel - 1) // 2
        xp = torch.cat([x[:, :1].repeat(1, pad), x, x[:, -1:].repeat(1, pad)], dim=1)
        trend = self.avg(xp.unsqueeze(1)).squeeze(1)
        trend = trend[:, : x.size(1)]
        seasonal = x - trend
        return trend, seasonal


class ApartElectricFormer(nn.Module):
    """
    Proposed model: a decomposition-LINEAR backbone (data-efficient, strong on
    short series) refined by a GATED attention correction that injects the
    electrical power-quality covariates and apartment periodicity.

        base       = Linear(trend) + Linear(seasonal)          # DLinear-style core
        correction = g * AttnBranch(patches, hour/dow emb, cov-fusion)
        y_hat      = base + tanh(alpha) * correction           # alpha init 0

    The scalar gate ``alpha`` starts at 0, so training begins as a pure linear
    forecaster (a very hard-to-beat floor on ~200 samples) and the Transformer
    only earns influence where it demonstrably lowers validation error.
    Every component is a toggle, giving a clean ablation study.
    """
    def __init__(self, n_feat, seq_len, out_dim, target_idx=0,
                 cov_slice=(1, 8), d_model=40, heads=4, layers=2, ff=80,
                 dropout=0.15, patch_len=6, stride=6, decomp_kernel=25,
                 use_decomp=True, use_patch=True, use_periodic=True,
                 use_covfuse=True, use_attn=True, use_clim=True, use_featmlp=True,
                 n_static=0):
        super().__init__()
        self.seq_len = seq_len
        self.target_idx = target_idx
        self.cov_lo, self.cov_hi = cov_slice
        self.use_decomp = use_decomp
        self.use_patch = use_patch
        self.use_periodic = use_periodic
        self.use_covfuse = use_covfuse
        self.use_attn = use_attn
        self.use_clim = use_clim
        self.use_featmlp = use_featmlp and n_static > 0
        self.out_dim = out_dim
        self.d_model = d_model

        # ---- static exogenous-feature fusion branch (covariate + calendar
        # summaries) - supplies the nonlinear power-quality / calendar signal
        # that lifts the multi-step (day-ahead) horizon. ----
        if self.use_featmlp:
            h = 64
            self.feat_mlp = nn.Sequential(
                nn.Linear(n_static, h), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(h, out_dim))
            nn.init.zeros_(self.feat_mlp[-1].weight); nn.init.zeros_(self.feat_mlp[-1].bias)

        self.decomp = _SeriesDecomp(decomp_kernel)

        # ---- hour-of-day climatology anchor (learned mean scaled load / hour) ----
        # keyed to the TARGET hours; encodes the dominant daily cycle in a very
        # data-efficient way (this is what lets day-ahead align to hour-of-day).
        self.hour_profile = nn.Embedding(24, 1)
        self.dow_profile = nn.Embedding(7, 1)
        nn.init.zeros_(self.hour_profile.weight)
        nn.init.zeros_(self.dow_profile.weight)

        # ---- linear backbone (DLinear-style) on the DESEASONALISED load ----
        self.lin_trend = nn.Linear(seq_len, out_dim)
        self.lin_seasonal = nn.Linear(seq_len, out_dim)
        for lin in (self.lin_trend, self.lin_seasonal):
            nn.init.normal_(lin.weight, std=1.0 / seq_len)
            nn.init.zeros_(lin.bias)

        # ---- attention correction branch ----
        if use_attn:
            in_ch = 2 if use_decomp else 1
            if use_patch:
                self.patch_len, self.stride = patch_len, stride
                self.n_patch = (seq_len - patch_len) // stride + 1
                self.patch_embed = nn.Linear(patch_len * in_ch, d_model)
                n_tokens = self.n_patch
            else:
                self.token_embed = nn.Linear(in_ch, d_model)
                n_tokens = seq_len
            self.n_tokens = n_tokens

            self.pos = nn.Parameter(torch.zeros(1, n_tokens, d_model))
            nn.init.trunc_normal_(self.pos, std=0.02)
            if use_periodic:
                self.hour_emb = nn.Embedding(24, d_model)
                self.dow_emb = nn.Embedding(7, d_model)

            n_cov = self.cov_hi - self.cov_lo
            if use_covfuse and n_cov > 0:
                self.cov_gru = nn.GRU(n_cov, d_model, batch_first=True)
                self.gate = nn.Sequential(nn.Linear(2 * d_model, d_model), nn.Sigmoid())
                self.cov_proj = nn.Linear(d_model, d_model)

            enc = nn.TransformerEncoderLayer(d_model, heads, ff, dropout,
                                             batch_first=True, activation="gelu")
            self.encoder = nn.TransformerEncoder(enc, layers)
            self.norm = nn.LayerNorm(d_model)
            self.corr_head = nn.Sequential(nn.Flatten(), nn.Dropout(dropout),
                                           nn.Linear(d_model * n_tokens, out_dim))
            self.alpha = nn.Parameter(torch.zeros(1))     # gate on the correction

    def _patchify(self, sig):
        B, L, C = sig.shape
        return torch.stack([sig[:, i * self.stride:i * self.stride + self.patch_len, :].reshape(B, -1)
                            for i in range(self.n_patch)], dim=1)

    def forward(self, x, hid=None, dow=None, sf=None):
        load = x[:, :, self.target_idx]                       # (B, L) scaled
        out_dim = self.out_dim

        # ---- climatology anchor keyed to TARGET hours (derived from hid/dow) ----
        if self.use_clim and hid is not None:
            input_clim = self.hour_profile(hid).squeeze(-1) + self.dow_profile(dow).squeeze(-1)
            k = torch.arange(out_dim, device=x.device).unsqueeze(0)       # (1, out_dim)
            steps = hid[:, -1:].float() + 1.0 + k                         # absolute hour offset
            tgt_h = (steps.long()) % 24
            tgt_d = (dow[:, -1:] + (steps.long() // 24)) % 7
            clim_out = self.hour_profile(tgt_h).squeeze(-1) + self.dow_profile(tgt_d).squeeze(-1)
            dev = load - input_clim                                       # deseasonalised load
        else:
            clim_out = torch.zeros(load.size(0), out_dim, device=x.device)
            dev = load

        if self.use_decomp:
            trend, seasonal = self.decomp(dev)
        else:
            trend, seasonal = dev, torch.zeros_like(dev)

        base = clim_out + self.lin_trend(trend) + self.lin_seasonal(seasonal)   # (B, out_dim)
        if self.use_featmlp and sf is not None:
            base = base + self.feat_mlp(sf)
        if not self.use_attn:
            return base

        sig = torch.stack([trend, seasonal], dim=-1) if self.use_decomp else dev.unsqueeze(-1)
        tokens = self.patch_embed(self._patchify(sig)) if self.use_patch else self.token_embed(sig)

        z = tokens + self.pos
        if self.use_periodic and hid is not None:
            per = self.hour_emb(hid) + self.dow_emb(dow)      # (B, L, d)
            if self.use_patch:
                per = torch.stack([per[:, i * self.stride:i * self.stride + self.patch_len, :].mean(1)
                                   for i in range(self.n_patch)], dim=1)
            z = z + per

        if self.use_covfuse and hasattr(self, "cov_gru"):
            cov = x[:, :, self.cov_lo:self.cov_hi]
            _, hcov = self.cov_gru(cov)
            ctx = self.cov_proj(hcov[-1]).unsqueeze(1).expand(-1, z.size(1), -1)
            g = self.gate(torch.cat([z, ctx], dim=-1))
            z = g * z + (1 - g) * ctx

        z = self.norm(self.encoder(z))
        corr = self.corr_head(z)
        return base + torch.tanh(self.alpha) * corr


# --------------------------------------------------------------------------- #
# Training loop (shared)
# --------------------------------------------------------------------------- #
def _t(a, device, dtype=torch.float32):
    return None if a is None else torch.tensor(a, dtype=dtype, device=device)


def train_torch(model, Xtr, ytr, hidtr, dowtr, Xva, yva, hidva, dowva,
                device, epochs=200, patience=25, lr=1e-3, wd=1e-4,
                batch=32, seed=0, verbose=False, sftr=None, sfva=None):
    torch.manual_seed(seed); np.random.seed(seed)
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=8)
    loss_fn = nn.SmoothL1Loss()

    Xtr = _t(Xtr, device)
    ytr = _t(ytr, device)
    hidtr = _t(hidtr, device, torch.long)
    dowtr = _t(dowtr, device, torch.long)
    Xva = _t(Xva, device)
    yva = _t(yva, device)
    hidva = _t(hidva, device, torch.long)
    dowva = _t(dowva, device, torch.long)
    sftr = _t(sftr, device); sfva = _t(sfva, device)
    if ytr.ndim == 1:
        ytr = ytr.unsqueeze(1); yva = yva.unsqueeze(1)

    n = Xtr.size(0)
    best, best_state, wait = float("inf"), None, 0
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            pred = model(Xtr[idx], hidtr[idx], dowtr[idx],
                         None if sftr is None else sftr[idx])
            loss = loss_fn(pred, ytr[idx])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            vpred = model(Xva, hidva, dowva, sfva)
            vloss = loss_fn(vpred, yva).item()
        sched.step(vloss)
        if vloss < best - 1e-6:
            best, wait = vloss, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


@torch.no_grad()
def predict_torch(model, X, hid, dow, device, sf=None):
    model.eval()
    X = torch.tensor(X, dtype=torch.float32, device=device)
    hid = torch.tensor(hid, dtype=torch.long, device=device)
    dow = torch.tensor(dow, dtype=torch.long, device=device)
    sf = None if sf is None else torch.tensor(sf, dtype=torch.float32, device=device)
    out = model(X, hid, dow, sf).cpu().numpy()
    return out
