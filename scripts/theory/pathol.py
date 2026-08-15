"""Are the trips with a runaway E[n] the same trips whose log Z is unstable?

Two loose ends met: run11's mean E[n] is 13.4 against a median of 7.5 with one trip at
116.4 (nmax is 120), and an earlier seed study found a worst-trip log Z sd of 2.28 nats
against a median of 0.041.  Both point at a small set of trips behaving differently from
the rest.  This checks whether it is the SAME set, and what those trips have in common.
"""
import sys, numpy as np, torch
sys.path.insert(0, '../v3'); torch.set_default_dtype(torch.float64)
from data import build; from features import Features; from fit import Batcher
from ragged import RaggedModel

D = build(); F = Features(int(D["n_item"]), int(D["n_store"]), 712); Bt = Batcher(D, F, 120)
m = RaggedModel(J=int(D["n_item"]), N=int(D["n_user"]), C=int(D["n_cat"]), K=32, Kz=12,
                nmax=120, R=23, S=int(D["n_store"]), Kp=8)
m.load_state_dict(torch.load('../../out/v3_run11.pt', map_location='cpu'))
m.double().eval()
va = np.flatnonzero(D["trip_split"] == 1)[:64]

En, ess, sd, nslot = [], [], [], []
for i in range(0, len(va), 16):
    ix, ctx, lctx, hh, LI, LT, LC, LU = Bt.make(va[i:i+16])
    m.house, m.ctx = hh, ctx
    g = torch.Generator().manual_seed(7)
    p = m.size_dist(ix, n_draws=64, generator=g)
    En.append((p * torch.arange(1, p.shape[1]+1, dtype=p.dtype)).sum(1).numpy())
    lzs = []
    for s in range(8):
        g = torch.Generator().manual_seed(100 + s)
        lz, e = m.log_Z(ix, n_draws=16, generator=g, return_ess=True, drop_empty=True)
        lzs.append(lz.detach().numpy())
        if s == 0:
            ess.append(e.detach().numpy())
    sd.append(np.std(np.stack(lzs), axis=0))
    nslot.append(np.bincount(ix.item_trip.numpy(), minlength=ix.B))
En, ess, sd, nslot = map(np.concatenate, (En, ess, sd, nslot))
obs = D["trip_nlines"][va]

print(f"\nE[n]:  median {np.median(En):.2f}  p90 {np.percentile(En,90):.2f}  max {En.max():.2f}")
print(f"logZ sd: median {np.median(sd):.4f}  max {sd.max():.4f}")
hi = En > np.percentile(En, 90)
print(f"\ncorrelation E[n] vs logZ sd     {np.corrcoef(En, sd)[0,1]:+.3f}")
print(f"correlation E[n] vs assortment  {np.corrcoef(En, nslot)[0,1]:+.3f}")
print(f"correlation E[n] vs observed n  {np.corrcoef(En, obs)[0,1]:+.3f}")
print(f"\nlogZ sd  in top-decile E[n] {sd[hi].mean():.4f}   rest {sd[~hi].mean():.4f}")
print(f"ESS      in top-decile E[n] {ess[hi].mean():.4f}   rest {ess[~hi].mean():.4f}")
print(f"assortment top-decile {nslot[hi].mean():8.0f}   rest {nslot[~hi].mean():8.0f}")
o = np.argsort(-En)[:6]
print(f"\n{'trip':>6}{'E[n]':>9}{'obs n':>7}{'slots':>8}{'logZ sd':>10}{'ESS':>8}")
for k in o:
    print(f"{int(va[k]):6d}{En[k]:9.2f}{int(obs[k]):7d}{int(nslot[k]):8d}{sd[k]:10.4f}{ess[k]:8.3f}")
