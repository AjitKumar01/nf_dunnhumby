"""Two things one seed and two summary statistics cannot tell you.

A. The SIZE DISTRIBUTION, not its mean and median.  Every calibration number reported so
   far has been E[n] and its median.  A model can match both and still put its mass in the
   wrong places, so here the simulated basket sizes are compared to the observed ones
   decile by decile, on the same test trips.

B. The PRICE EFFECT, averaged over seeds.  A single 3-week rollout showed a 10% cut
   producing 5% FEWER lines -- the wrong sign -- when the elasticity implies about +1.2%.
   An effect that small is below the noise of one rollout, so it is repeated and reported
   with a standard error.
"""
import sys, math, time, numpy as np, torch
sys.path.insert(0, '../v3'); torch.set_default_dtype(torch.float64)
from data import build; from features import Features; from fit import Batcher
from ragged import RaggedModel

D = build(); F = Features(int(D["n_item"]), int(D["n_store"]), 712); Bt = Batcher(D, F, 120)
J, N, C, S = (int(D[k]) for k in ("n_item", "n_user", "n_cat", "n_store"))
m = RaggedModel(J=J, N=N, C=C, K=32, Kz=12, nmax=120, R=23, S=S, Kp=8)
m.load_state_dict(torch.load('../../out/v3_run22_cal.pt', map_location='cpu'))
m.double().eval()

tst = np.flatnonzero(D["trip_split"] == 2)[:128]
obs = D["trip_nlines"][tst]

# ---- A. the whole distribution -------------------------------------------------------
sim = []
t0 = time.time()
for rep in range(8):
    g = torch.Generator().manual_seed(100 + rep)
    for k in range(0, len(tst), 32):
        ix, ctx, lctx, hh, *_ = Bt.make(tst[k:k+32])
        m.house, m.ctx = hh, ctx
        sim.extend(len(b) for b in m.sample(ix, n_draws=16, generator=g))
sim = np.array(sim)
qs = [10, 25, 50, 75, 90, 95, 99]
print(f"A. basket-size distribution, {len(tst)} test trips x 8 replications "
      f"({time.time()-t0:.0f}s)")
print("   decile      " + "".join(f"{q:>7}%" for q in qs) + "     mean")
print("   observed    " + "".join(f"{np.percentile(obs,q):8.1f}" for q in qs)
      + f"{obs.mean():9.2f}")
print("   simulated   " + "".join(f"{np.percentile(sim,q):8.1f}" for q in qs)
      + f"{sim.mean():9.2f}")
print("   ratio       " + "".join(
    f"{np.percentile(sim,q)/max(np.percentile(obs,q),1e-9):8.2f}" for q in qs)
    + f"{sim.mean()/obs.mean():9.2f}")
print(f"   share of baskets with 1-3 items:  observed {(obs<=3).mean():.1%}   "
      f"simulated {(sim<=3).mean():.1%}")
print(f"   share with more than 20 items:    observed {(obs>20).mean():.1%}   "
      f"simulated {(sim>20).mean():.1%}")

# ---- B. the price effect, over seeds --------------------------------------------------
print(f"\nB. price effect on basket size, {len(tst)} test trips x 10 seeds")
res = {}
for dlp, tag in ((0.0, "no change"), (math.log(0.90), "10% cut"),
                 (math.log(1.10), "10% rise")):
    tot = []
    for rep in range(10):
        g = torch.Generator().manual_seed(500 + rep)
        n = 0
        for k in range(0, len(tst), 32):
            ix, ctx, lctx, hh, *_ = Bt.make(tst[k:k+32])
            c = dict(ctx); c["dlp"] = ctx["dlp"] + dlp
            m.house, m.ctx = hh, c
            n += sum(len(b) for b in m.sample(ix, n_draws=16, generator=g))
        tot.append(n)
    tot = np.array(tot, float)
    res[tag] = tot
    print(f"   {tag:10s} lines {tot.mean():8.1f} +/- {tot.std(ddof=1)/math.sqrt(10):5.1f}")
base = res["no change"].mean()
for tag in ("10% cut", "10% rise"):
    d = res[tag] - res["no change"]
    se = d.std(ddof=1) / math.sqrt(10)
    print(f"   {tag:10s} change {100*d.mean()/base:+6.2f}% +/- {100*se/base:4.2f}%"
          f"   (elasticity implies {'+1.2' if 'cut' in tag else '-1.1'}%)")
