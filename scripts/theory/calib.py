"""Is the model's basket-size law calibrated?  Chunked over trips: the convolution buffer
is [draws, trips * categories, R+1], which OOMs the machine at 256 trips x 256 draws."""
import sys, numpy as np, torch
sys.path.insert(0, '../v3'); torch.set_default_dtype(torch.float64)
from data import build; from features import Features; from fit import Batcher
from ragged import RaggedModel

D = build(); F = Features(int(D["n_item"]), int(D["n_store"]), 712); Bt = Batcher(D, F, 120)
va = np.flatnonzero(D["trip_split"] == 1)[:128]
obs = D["trip_nlines"][va]
tr = np.flatnonzero(D["trip_split"] == 0)
emp = np.bincount(np.clip(D["trip_nlines"][tr], 0, 120), minlength=121).astype(float)
emp /= emp.sum()

for ck in ("run9", "run11"):
    m = RaggedModel(J=int(D["n_item"]), N=int(D["n_user"]), C=int(D["n_cat"]), K=32, Kz=12,
                    nmax=120, R=23, S=int(D["n_store"]), Kp=8)
    m.load_state_dict(torch.load(f'../../out/v3_{ck}.pt', map_location='cpu'))
    m.double().eval()
    E, P = [], np.zeros(121)
    for i in range(0, len(va), 16):
        ix, ctx, lctx, hh, LI, LT, LC, LU = Bt.make(va[i:i+16])
        m.house, m.ctx = hh, ctx
        g = torch.Generator().manual_seed(7)
        p = m.size_dist(ix, n_draws=64, generator=g)          # [B, n], index i -> size i+1
        E.append((p * torch.arange(1, p.shape[1]+1, dtype=p.dtype)).sum(1).numpy())
        P[1:p.shape[1]+1] += p.sum(0).numpy()
    E = np.concatenate(E); P /= P.sum()
    print(f"\n=== {ck} ===")
    print(f"  E[n]  mean {E.mean():7.3f}  median {np.median(E):7.3f}  "
          f"p90 {np.percentile(E,90):7.3f}  max {E.max():7.3f}")
    print(f"  observed   mean {obs.mean():7.3f}  median {np.median(obs):7.3f}  "
          f"p90 {np.percentile(obs,90):7.3f}  max {obs.max():7.3f}")
    print(f"  model P(n) mass n<=10: {P[1:11].sum():.3f}   empirical {emp[1:11].sum():.3f}")
    print(f"  model P(n) mass n>20 : {P[21:].sum():.3f}   empirical {emp[21:].sum():.3f}")
    print("   n:  " + "".join(f"{k:7d}" for k in (1,2,3,5,8,12,20,30,50)))
    print("  mdl: " + "".join(f"{P[k]:7.4f}" for k in (1,2,3,5,8,12,20,30,50)))
    print("  emp: " + "".join(f"{emp[k]:7.4f}" for k in (1,2,3,5,8,12,20,30,50)))
