"""
Stage 41 -- Could this model target coupons, and can its baskets train a policy?

PART A -- TARGETING.  The model gives a per-(household, product) price slope
g_ij = gamma_i.beta_j.  Targeting is only worth anything if that predicted heterogeneity
matches REAL heterogeneity out of sample.  The test:

  1. rank households by predicted sensitivity to a product, using TRAIN-fitted parameters
  2. on HELD-OUT weeks, measure each group's ACTUAL purchase response to price deviations
  3. if the predicted-sensitive group responds more, targeting has value; if the groups
     respond alike, g_ij's household dimension is noise and only the product dimension is
     usable

Reported as the actual elasticity by predicted-sensitivity quintile, and as the implied
gain from targeting the top quintile instead of a random household.

PART B -- MARKOV POLICIES.  A pricing policy learned on generated baskets needs the
generator to be a Markov decision process: a state that summarises history, a transition
that depends only on (state, action), and a reward.  The model has exactly one dynamic
state -- the purchase-recency vector x_ijt -- so the question is whether the generated
data reproduces the real transition, i.e. whether P(buy | days since last purchase) in
generated baskets matches the real curve.  That curve IS the transition kernel of the only
state variable, so if it is wrong every long-horizon rollout drifts.

Writes out/targeting_<label>.json.
"""
import argparse
import importlib
import json
import os

import numpy as np
import pandas as pd
import torch

nb = importlib.import_module("27_nested_basket")
cf = importlib.import_module("28_nested_counterfactual")

HERE = os.path.dirname(os.path.abspath(__file__))
IN = os.path.join(HERE, "..", "basket_input")
OUT = os.path.join(HERE, "..", "out")


def log(m):
    print(f"[41] {m}", flush=True)


def main(a):
    dev = torch.device("cpu")
    d = nb.NestedData(IN, device=dev)
    m, _ = cf.load(a.label, d, dev)
    G = (m.gamma.detach() @ m.beta.detach().T).numpy()          # [N, J]
    res = {"label": a.label}

    # ---------------------------------------------------------------- PART A
    log("")
    log("A. TARGETING -- does predicted sensitivity match real held-out response?")
    sp = d.splits["test"]
    lp = d.log_price_dev.numpy()

    # Real response: for each (household, product) with enough held-out exposure, regress
    # the purchase indicator on the price deviation across the weeks the household shopped.
    tr = d.splits["train"]
    bought_tr = np.zeros((d.N, d.J), dtype=bool)
    bought_tr[tr["user"], tr["item"]] = True

    # candidate (household, product) pairs: bought at least `min_tr` times in train, so the
    # household plausibly considers the product at all
    cnt = np.zeros((d.N, d.J), dtype=np.int32)
    np.add.at(cnt, (tr["user"], tr["item"]), 1)
    hh, pj = np.nonzero(cnt >= a.min_train)
    log(f"   {len(hh):,} (household, product) pairs bought >= {a.min_train}x in train")

    # held-out trips per household, and what they bought
    trips = {}
    for i in range(sp["n_baskets"]):
        u = sp["user"][sp["starts"][i]]
        trips.setdefault(int(u), []).append(i)
    got = set(zip(sp["user"].tolist(), sp["item"].tolist()))

    rows = []
    keep = np.random.default_rng(a.seed).choice(
        len(hh), size=min(a.n_pairs, len(hh)), replace=False)
    for k in keep:
        u, j = int(hh[k]), int(pj[k])
        tl = trips.get(u, [])
        if len(tl) < a.min_trips:
            continue
        days = np.array([sp["day"][sp["starts"][i]] for i in tl])
        dev_ = lp[j, days]
        if dev_.std() < 1e-6:
            continue
        y = np.array([1.0 if int(sp["item"][sp["starts"][i]:sp["ends"][i]].tolist().count(j) > 0)
                      else 0.0 for i in tl]) if False else None
        # purchased on that trip?
        y = np.zeros(len(tl))
        for q, i in enumerate(tl):
            y[q] = float(j in sp["item"][sp["starts"][i]:sp["ends"][i]])
        if y.sum() == 0 or y.sum() == len(y):
            continue
        # within-pair slope of purchase on price deviation
        xd = dev_ - dev_.mean()
        b = float((xd * (y - y.mean())).sum() / max((xd ** 2).sum(), 1e-12))
        rows.append((u, j, b, G[u, j], len(tl)))
    R = pd.DataFrame(rows, columns=["u", "j", "real_slope", "pred_g", "n_trips"])
    log(f"   {len(R):,} pairs with usable held-out price variation "
        f"(>= {a.min_trips} trips, non-constant price, mixed outcome)")

    if len(R) > 200:
        from scipy.stats import spearmanr
        R["q"] = pd.qcut(R.pred_g, 5, labels=False, duplicates="drop")
        log("")
        log(f"   {'predicted-sensitivity quintile':34s} {'n':>6s} "
            f"{'mean pred g':>12s} {'REAL slope':>11s}")
        tab = []
        for q, g in R.groupby("q"):
            log(f"   {'Q'+str(int(q)+1)+(' (least sensitive)' if q==0 else ' (most sensitive)' if q==R.q.max() else ''):34s}"
                f" {len(g):6d} {g.pred_g.mean():12.3f} {g.real_slope.mean():11.4f}")
            tab.append({"quintile": int(q) + 1, "n": int(len(g)),
                        "pred_g": float(g.pred_g.mean()),
                        "real_slope": float(g.real_slope.mean())})
        rho = spearmanr(R.pred_g, R.real_slope)
        log("")
        log(f"   spearman(predicted g, real held-out slope) = {rho.statistic:+.4f} "
            f"(p={rho.pvalue:.3g}, n={len(R):,})")
        log("   a NEGATIVE real slope means the household buys less when the price is high,")
        log("   so targeting works if the most-sensitive quintile has the most negative slope")
        res["targeting"] = {"quintiles": tab, "spearman": float(rho.statistic),
                            "p": float(rho.pvalue), "n": int(len(R))}

        # how much of the predicted signal is household vs product?
        rp = spearmanr(R.groupby("j").pred_g.mean(), R.groupby("j").real_slope.mean())
        log(f"   same, aggregated to PRODUCTS: {rp.statistic:+.4f} "
            f"(n={R.j.nunique():,} products)")
        ru = spearmanr(R.groupby("u").pred_g.mean(), R.groupby("u").real_slope.mean())
        log(f"   same, aggregated to HOUSEHOLDS: {ru.statistic:+.4f} "
            f"(n={R.u.nunique():,} households)")
        res["targeting"]["spearman_products"] = float(rp.statistic)
        res["targeting"]["spearman_households"] = float(ru.statistic)

    # ---------------------------------------------------------------- PART B
    log("")
    log("B. MARKOV -- does generated data reproduce the recency transition?")
    log("   x_ijt (days since last purchase) is the model's ONLY dynamic state, so")
    log("   P(buy | recency) is the transition kernel any policy would roll out.")
    gen = cf.generate_baskets(m, d, dev, n_trips=a.n_trips, seed=a.seed,
                              sweeps=4, use_ctx=True, with_units=False)
    log(f"   generated {len(gen):,} baskets")

    # real: for held-out trips, days since the household last bought the sub-commodity
    def recency_curve(user_arr, item_arr, day_arr, tag):
        tau = d.state(user_arr, item_arr, day_arr)[:, 3]   # log(1+tau)/log(100)
        bins = np.array([0, .25, .4, .5, .6, .7, .8, 1.01])
        idx = np.digitize(tau, bins) - 1
        out = []
        for b in range(len(bins) - 1):
            msk = idx == b
            if msk.sum() > 50:
                out.append((float(bins[b]), int(msk.sum())))
        return tau

    real_rows = np.concatenate([np.arange(sp["starts"][i], sp["ends"][i])
                                for i in range(min(4000, sp["n_baskets"]))])
    t_real = d.state(sp["user"][real_rows], sp["item"][real_rows],
                     sp["day"][real_rows])[:, 3]
    # gen.trips[k] is the trip the k-th generated basket came from.  Indexing by k
    # instead compares a generated basket against a DIFFERENT household's history, which
    # makes an accurate generator look badly wrong.
    tp = gen.trips
    gi = np.concatenate([np.asarray(b) for b in gen if len(b)])
    gu = np.concatenate([np.full(len(b), int(sp["user"][sp["starts"][tp[k]]]))
                         for k, b in enumerate(gen) if len(b)])
    gd = np.concatenate([np.full(len(b), int(sp["day"][sp["starts"][tp[k]]]))
                         for k, b in enumerate(gen) if len(b)])
    t_gen = d.state(gu, gi, gd)[:, 3]
    qs = [0, .1, .25, .5, .75, .9, 1.0]
    log("")
    log(f"   {'quantile of log(1+tau)/log100':32s} {'real':>8s} {'generated':>10s}")
    rq, gq = [], []
    for q in qs:
        r, g = float(np.quantile(t_real, q)), float(np.quantile(t_gen, q))
        rq.append(r); gq.append(g)
        log(f"   {('q'+str(int(q*100))):32s} {r:8.4f} {g:10.4f}")
    never_r = float((t_real == 0).mean()); never_g = float((t_gen == 0).mean())
    log(f"   {'share never bought before':32s} {never_r:8.2%} {never_g:10.2%}")
    res["markov"] = {"recency_quantiles_real": rq, "recency_quantiles_gen": gq,
                     "never_real": never_r, "never_gen": never_g,
                     "mean_real": float(t_real.mean()), "mean_gen": float(t_gen.mean())}
    log(f"   mean recency feature: real {t_real.mean():.4f}  generated {t_gen.mean():.4f}")

    with open(os.path.join(OUT, f"targeting_{a.label}.json"), "w") as f:
        json.dump(res, f, indent=2)
    log("")
    log(f"wrote out/targeting_{a.label}.json")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--label", default="spec_nested")
    p.add_argument("--min-train", type=int, default=3)
    p.add_argument("--min-trips", type=int, default=8)
    p.add_argument("--n-pairs", type=int, default=60000)
    p.add_argument("--n-trips", type=int, default=6000)
    p.add_argument("--seed", type=int, default=0)
    main(p.parse_args())
