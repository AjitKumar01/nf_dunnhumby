"""run22 vs run23, over seeds, so the trade-off is conclusive rather than suggestive.

The single-seed reads were: elasticity -0.114 vs -0.120 against a data value of -0.121, and
E[n] 11.60 vs 10.07 against 6.83.  Those gaps are small enough that one seed cannot say
whether they are real.  Both are re-measured over 8 independent seeds with standard errors,
alongside the generated basket size under a price cut and a price rise.
"""
import sys, math, numpy as np, torch
sys.path.insert(0, '../v3'); torch.set_default_dtype(torch.float64)
from data import build; from features import Features; from fit import Batcher
from ragged import RaggedModel

D = build(); F = Features(int(D["n_item"]), int(D["n_store"]), 712); Bt = Batcher(D, F, 120)
J, N, C, S = (int(D[k]) for k in ("n_item", "n_user", "n_cat", "n_store"))
tst = np.flatnonzero(D["trip_split"] == 2)[:96]
obs = D["trip_nlines"][tst]
SEEDS = 8

def moments(m, dlp, seed):
    out = []
    for k in range(0, len(tst), 32):
        ix, ctx, lctx, hh, *_ = Bt.make(tst[k:k+32])
        m.house, m.ctx = hh, ctx
        g = torch.Generator().manual_seed(seed)
        _, zh = m.size_dist(ix, n_draws=8, generator=g, return_mode=True)
        c = dict(ctx); c["dlp"] = ctx["dlp"] + dlp
        m.ctx = c
        g = torch.Generator().manual_seed(seed)
        e, _ = m.size_moments(ix, n_draws=64, generator=g, z_fixed=zh)
        out.append(e.numpy())
    return np.concatenate(out).mean()

def gen_lines(m, dlp, seed):
    n = 0
    for k in range(0, len(tst), 32):
        ix, ctx, lctx, hh, *_ = Bt.make(tst[k:k+32])
        c = dict(ctx); c["dlp"] = ctx["dlp"] + dlp
        m.house, m.ctx = hh, c
        g = torch.Generator().manual_seed(seed)
        n += sum(len(b) for b in m.sample(ix, n_draws=16, generator=g))
    return n

def se(x):
    x = np.asarray(x, float)
    return x.mean(), x.std(ddof=1) / math.sqrt(len(x))

print(f"{len(tst)} test trips, {SEEDS} seeds.  observed mean basket {obs.mean():.2f}, "
      f"lines {obs.sum()}")
for tag, ck in (("run22+recal", 'v3_run22_cal.pt'), ("run23+recal", 'v3_run23_cal.pt')):
    m = RaggedModel(J=J, N=N, C=C, K=32, Kz=12, nmax=120, R=23, S=S, Kp=8)
    m.load_state_dict(torch.load(f'../../out/{ck}', map_location='cpu')); m.double().eval()
    en, el, gl = [], [], {0.0: [], math.log(0.90): [], math.log(1.10): []}
    for s in range(SEEDS):
        e0 = moments(m, 0.0, 10 + s)
        ep = moments(m, +0.05, 10 + s)
        em = moments(m, -0.05, 10 + s)
        en.append(e0)
        el.append((math.log(ep) - math.log(em)) / 0.10)
        for d in gl:
            gl[d].append(gen_lines(m, d, 200 + s))
    m_en, s_en = se(en); m_el, s_el = se(el)
    b_m, b_s = se(gl[0.0]); c_m, c_s = se(gl[math.log(0.90)]); r_m, r_s = se(gl[math.log(1.10)])
    print(f"\n{tag}")
    print(f"   E[n]        {m_en:6.2f} +/- {s_en:.2f}      (observed {obs.mean():.2f})")
    print(f"   elasticity  {m_el:+6.3f} +/- {s_el:.3f}     (data -0.121)")
    print(f"   lines base  {b_m:7.1f} +/- {b_s:.1f}     (observed {obs.sum()})")
    dc = np.array(gl[math.log(0.90)], float) - np.array(gl[0.0], float)
    dr = np.array(gl[math.log(1.10)], float) - np.array(gl[0.0], float)
    print(f"   10% cut     {100*dc.mean()/b_m:+6.2f}% +/- {100*dc.std(ddof=1)/math.sqrt(SEEDS)/b_m:.2f}%")
    print(f"   10% rise    {100*dr.mean()/b_m:+6.2f}% +/- {100*dr.std(ddof=1)/math.sqrt(SEEDS)/b_m:.2f}%")
