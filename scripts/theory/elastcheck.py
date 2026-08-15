"""Aggregate price elasticity of basket size, measured the same way on model and data.

The data's value, from regressing log(basket size) on the mean price deviation faced over
40,000 training trips, is -0.121.  Here the same quantity is taken from the model by finite
difference on E[n] under a uniform log-price shift, on TEST trips, with the proposal held
common between the two evaluations -- differencing two estimates on two different proposals
carries the proposal change along with the effect (that error was worth 15% earlier).
"""
import sys, math, numpy as np, torch
sys.path.insert(0, '../v3'); torch.set_default_dtype(torch.float64)
from data import build; from features import Features; from fit import Batcher
from ragged import RaggedModel

D = build(); F = Features(int(D["n_item"]), int(D["n_store"]), 712); Bt = Batcher(D, F, 120)
J, N, C, S = (int(D[k]) for k in ("n_item", "n_user", "n_cat", "n_store"))
tst = np.flatnonzero(D["trip_split"] == 2)[:96]

for tag, ck in (("run23 + recal", 'v3_run23_cal.pt'), ("run22 + recal", 'v3_run22_cal.pt')):
    m = RaggedModel(J=J, N=N, C=C, K=32, Kz=12, nmax=120, R=23, S=S, Kp=8)
    m.load_state_dict(torch.load(f'../../out/{ck}', map_location='cpu')); m.double().eval()
    E = {}
    for d in (0.0, -0.05, +0.05):
        tot = []
        for k in range(0, len(tst), 32):
            ix, ctx, lctx, hh, *_ = Bt.make(tst[k:k+32])
            m.house, m.ctx = hh, ctx
            g = torch.Generator().manual_seed(3)
            _, zh = m.size_dist(ix, n_draws=8, generator=g, return_mode=True)
            c = dict(ctx); c["dlp"] = ctx["dlp"] + d
            m.ctx = c
            g = torch.Generator().manual_seed(3)
            e, _ = m.size_moments(ix, n_draws=64, generator=g, z_fixed=zh)
            tot.append(e.numpy())
        E[d] = np.concatenate(tot).mean()
    el = (math.log(E[+0.05]) - math.log(E[-0.05])) / 0.10
    print(f"{tag:16s} E[n] {E[0.0]:6.2f}   d log E[n] / d log p = {el:+.3f}"
          f"   (data -0.121)")
