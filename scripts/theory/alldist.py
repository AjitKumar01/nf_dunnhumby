"""Size was never the only distribution that has to be right.

A generative retail model is used to produce baskets, and a policy sees every aspect of
them, not just how many lines they contain.  Five more comparisons against the same test
trips, all simulated-vs-observed:

  units per line        the shifted-NB quantity model, never checked against data
  categories per basket breadth of the trip -- 8 items from 8 aisles is a different shop
                        from 8 items from 2
  spend per basket      the environment's REWARD.  If this is wrong nothing a policy learns
                        transfers, however right the item probabilities are.
  item purchase rates   do the right products appear, at the right frequency
  pair co-occurrence    complementarity is the model's whole claim over the baselines, and
                        it has never been checked against observed co-purchase rates
"""
import sys, time, math, numpy as np, torch
sys.path.insert(0, '../v3'); torch.set_default_dtype(torch.float64)
from torch.nn.functional import softplus
from data import build; from features import Features; from fit import Batcher
from ragged import RaggedModel

D = build(); F = Features(int(D["n_item"]), int(D["n_store"]), 712); Bt = Batcher(D, F, 120)
J, N, C, S = (int(D[k]) for k in ("n_item", "n_user", "n_cat", "n_store"))
m = RaggedModel(J=J, N=N, C=C, K=32, Kz=12, nmax=120, R=23, S=S, Kp=8)
m.load_state_dict(torch.load('../../out/v3_run22_cal.pt', map_location='cpu'))
m.double().eval()
logp = torch.from_numpy(np.load('../../basket_input/log_price.npy').astype(np.float64))
cat_of = np.zeros(J, np.int64)
lp, li, lc, lu = D["line_ptr"], D["line_item"], D["line_cat"], D["line_units"]
cat_of[li] = lc

tst = np.flatnonzero(D["trip_split"] == 2)[:128]
day = D["trip_day"]

# ---- observed ------------------------------------------------------------------------
o_units, o_cats, o_spend, o_items, o_pairs = [], [], [], [], {}
for t in tst:
    a, b = int(lp[t]), int(lp[t+1]); d = int(day[t])
    it, u = li[a:b], lu[a:b]
    if len(it) == 0: continue
    o_units.append(u.sum()/len(it)); o_cats.append(len(set(cat_of[it].tolist())))
    o_spend.append(float((np.exp(logp[it, d].numpy())*u).sum()))
    o_items.extend(it.tolist())
    s = sorted(set(int(x) for x in it))
    for i in range(len(s)):
        for j in range(i+1, len(s)):
            o_pairs[(s[i], s[j])] = o_pairs.get((s[i], s[j]), 0) + 1

# ---- simulated -----------------------------------------------------------------------
REP = 8
s_units, s_cats, s_spend, s_items, s_pairs = [], [], [], [], {}
t0 = time.time()
r_par = softplus(m.log_r).detach() + 1e-6
for rep in range(REP):
    g = torch.Generator().manual_seed(700+rep)
    for k in range(0, len(tst), 32):
        sub = tst[k:k+32]
        ix, ctx, lctx, hh, *_ = Bt.make(sub)
        m.house, m.ctx = hh, ctx
        for bi, items in enumerate(m.sample(ix, n_draws=16, generator=g)):
            if not items: continue
            it = torch.as_tensor(items, dtype=torch.long)
            d = int(day[sub[bi]]); h = int(D["trip_user"][sub[bi]])
            z = m.a_q[it].detach()
            mu = torch.exp(z.clamp(-6.0, 4.0))
            lam = torch._standard_gamma(r_par.expand(mu.shape)) * mu / r_par
            u = 1.0 + torch.poisson(lam, generator=g)
            s_units.append(float(u.sum())/len(it))
            s_cats.append(len(set(cat_of[items].tolist())))
            s_spend.append(float((torch.exp(logp[it, d])*u).sum()))
            s_items.extend(items)
            for i in range(len(items)):
                for j in range(i+1, len(items)):
                    key = (items[i], items[j])
                    s_pairs[key] = s_pairs.get(key, 0) + 1
print(f"{len(tst)} test trips x {REP} reps  ({time.time()-t0:.0f}s)")

def cmp(name, o, s, fmt="{:7.2f}"):
    o, s = np.array(o, float), np.array(s, float)
    row = lambda a: "".join(fmt.format(np.percentile(a, q)) for q in (25, 50, 75, 90))
    print(f"\n{name}")
    print(f"   observed   {row(o)}   mean {o.mean():7.2f}")
    print(f"   simulated  {row(s)}   mean {s.mean():7.2f}   ratio {s.mean()/o.mean():.2f}")

print("\n                    p25    p50    p75    p90")
cmp("units per line", o_units, s_units)
cmp("categories per basket", o_cats, s_cats)
cmp("spend per basket (REWARD)", o_spend, s_spend)

# item rates
cnt_o = np.bincount(np.array(o_items), minlength=J).astype(float)/len(tst)
cnt_s = np.bincount(np.array(s_items), minlength=J).astype(float)/(len(tst)*REP)
top = np.argsort(-cnt_o)[:200]
print(f"\nitem purchase rate, top 200 products")
print(f"   correlation observed vs simulated  {np.corrcoef(cnt_o[top], cnt_s[top])[0,1]:.3f}")
print(f"   mean rate  observed {cnt_o[top].mean():.4f}   simulated {cnt_s[top].mean():.4f}"
      f"   ratio {cnt_s[top].mean()/cnt_o[top].mean():.2f}")
print(f"   products the model never generates: "
      f"{int((cnt_s[top]==0).sum())}/200")

# pair co-occurrence on the pairs actually observed
common = [k for k, v in sorted(o_pairs.items(), key=lambda x: -x[1])[:200]]
po = np.array([o_pairs[k]/len(tst) for k in common])
ps = np.array([s_pairs.get(k, 0)/(len(tst)*REP) for k in common])
print(f"\npair co-occurrence, 200 most common observed pairs")
print(f"   correlation  {np.corrcoef(po, ps)[0,1]:.3f}")
print(f"   mean rate  observed {po.mean():.5f}   simulated {ps.mean():.5f}"
      f"   ratio {ps.mean()/max(po.mean(),1e-12):.2f}")
print(f"   observed pairs the model never generates: {int((ps==0).sum())}/200")
