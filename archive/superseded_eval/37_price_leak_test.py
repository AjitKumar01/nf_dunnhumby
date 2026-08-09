"""
Stage 37 -- Is the margin an artefact of how prices are reconstructed?

dunnhumby ships no price file.  The price at (item, day) is the MEDIAN over that day's
purchase lines, so for a thin item-day it is built from the very transaction being
scored.  41.2% of held-out purchase lines are the SOLE observation of their item-day.

That creates a structural asymmetry in the choice set:

    true item       100.0% have a same-day price observation, by construction
    sampled decoy    43.1% do

and observed prices are more variable than carried-forward ones.  So some part of "the
bought item was cheaper than its decoys" could be the observation mechanism rather than
shopper response.  Neither the structural placebo nor the price-variation evaluation
separates these, because both compare models on the same feature.

Two tests that do.

  1. THICKNESS STRATA.  Score only held-out rows whose item-day price rests on many
     purchase lines.  There the true item's price is not self-derived -- one line moves a
     median over twenty barely at all -- so the asymmetry is weak.  If the margin over
     baselines survives, the mechanism is not driving it.

  2. MATCHED DECOYS.  Restrict the negatives to items whose price was ALSO directly
     observed that same day.  This removes the asymmetry by construction rather than by
     stratification: every candidate, true or false, has a freshly observed price.

Writes out/price_leak_test.json.
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
IN = os.path.join(HERE, "..", "..", os.environ.get("NF_BASKET_INPUT", "basket_input"))
DATA = os.path.join(HERE, "..", "..", "data")
OUT = os.path.join(HERE, "..", "..", "out")


def log(m):
    print(f"[37] {m}", flush=True)


def main(a):
    dev = torch.device(a.device)
    d = nb.NestedData(IN, device=dev)
    sp = d.splits["test"]
    rng = np.random.default_rng(a.seed)

    # how many purchase lines back each (item, day) price
    items = pd.read_parquet(os.path.join(IN, "items.parquet"))
    tx = pd.read_parquet(os.path.join(DATA, "tx.parquet"),
                         columns=["PRODUCT_ID", "DAY"])
    tx = tx.merge(items[["PRODUCT_ID", "item_id"]], on="PRODUCT_ID")
    cnt = tx.groupby(["item_id", "DAY"]).size()
    lines = np.zeros((d.J, d.log_price_dev.shape[1]), dtype=np.int32)
    ij = np.array(cnt.index.tolist())
    lines[ij[:, 0], ij[:, 1]] = cnt.to_numpy()
    log(f"price-support matrix built: {(lines > 0).mean():.1%} of item-days observed")

    bidx = rng.choice(sp["n_baskets"], size=min(a.n_baskets, sp["n_baskets"]),
                      replace=False)
    rows = np.concatenate([np.arange(sp["starts"][i], sp["ends"][i]) for i in bidx])
    user, item = sp["user"][rows], sp["item"][rows]
    day, week = sp["day"][rows], sp["week"][rows]
    store, rw = sp["store"][rows], sp["raw_week"][rows]
    B = len(rows)
    support = lines[item, day]
    log(f"{B:,} held-out rows; support median {np.median(support):.0f}, "
        f"{np.mean(support == 1):.1%} rest on a single line")

    # ---- household repeat-purchase, the strongest basket-free baseline, tuned once
    tr = d.splits["train"]
    Hm = np.zeros((d.N, d.J), dtype=np.float32)
    np.add.at(Hm, (tr["user"], tr["item"]), 1.0)
    Hm = np.log1p(Hm)

    def build(neg):
        cnp = np.concatenate([item[:, None], neg], axis=1)
        M = cnp.shape[1]
        cand = torch.as_tensor(cnp, device=dev)
        dr = np.repeat(day[:, None], M, 1); ur = np.repeat(user[:, None], M, 1)
        sr = np.repeat(store[:, None], M, 1); rr = np.repeat(rw[:, None], M, 1)
        st = torch.as_tensor(d.state(ur.ravel(), cnp.ravel(), dr.ravel()).reshape(
            B, M, nb.N_STATE_FEATURES), device=dev)
        av = d.carried[cand, torch.as_tensor(sr, device=dev)].clone()
        av[:, 0] = True
        return cnp, cand, dr, sr, rr, st, av

    def score_all(cnp, cand, dr, sr, rr, st, av, mask, tag):
        out = {}
        M = cnp.shape[1]
        s = torch.as_tensor(Hm[user[:, None], cnp], device=dev) * a.w_repeat
        s = s.masked_fill(~av, -1e9)
        lp = torch.log_softmax(s, 1)[:, 0].cpu().numpy()
        out["household repeat"] = float(lp[mask].mean())
        for lb in a.labels:
            if not os.path.exists(os.path.join(OUT, f"{lb}_nested.pt")):
                continue
            m, _ = cf.load(lb, d, dev)
            dl = d.log_price_dev[cand, torch.as_tensor(dr, device=dev)]
            if m.use_store and m.use_store_price:
                dl = dl + d.store_dev(cnp.ravel(), sr.ravel(), rr.ravel()).reshape(B, M)
            A = m.alpha.detach()
            ctx = torch.zeros(B, m.K, device=dev)
            starts = np.concatenate([[0], np.cumsum([sp["ends"][i] - sp["starts"][i]
                                                    for i in bidx])])
            for bi, i in enumerate(bidx):
                r = np.arange(starts[bi], starts[bi + 1])
                its = item[r]
                if len(its) > 1:
                    tot = A[torch.as_tensor(its, device=dev)].sum(0)
                    for q, rr_ in enumerate(r):
                        ctx[rr_] = (tot - A[its[q]]) / (len(its) - 1)
            with torch.no_grad():
                u = m.item_utility(torch.as_tensor(user, device=dev), cand, ctx, dl, st,
                                   torch.as_tensor(week, device=dev),
                                   torch.as_tensor(store, device=dev))
            u = u.masked_fill(~av, -1e9)
            lp = torch.log_softmax(u, 1)[:, 0].cpu().numpy()
            out[lb] = float(lp[mask].mean())
        return out

    res = {"rows": int(B), "single_line_share": float(np.mean(support == 1))}

    # ---------- test 1: thickness strata, ordinary negatives
    log("")
    log("1. by how many purchase lines back the TRUE item's price")
    neg = rng.choice(d.J, size=(B, a.n_neg), p=d.neg_p).astype(np.int64)
    built = build(neg)
    strata = [("support = 1", support == 1), ("2-4", (support >= 2) & (support <= 4)),
              ("5-19", (support >= 5) & (support <= 19)), ("20+", support >= 20)]
    res["strata"] = {}
    for name, mask in strata:
        if mask.sum() < 200:
            continue
        o = score_all(*built, mask, name)
        res["strata"][name] = {"n": int(mask.sum()), **o}
        best = min(k for k in o if k != "household repeat")
        log(f"   {name:12s} n={mask.sum():7,}  " + "  ".join(
            f"{k}:{v:+.4f}" for k, v in o.items())
            + f"   margin {o[a.labels[0]] - o['household repeat']:+.3f}")

    # ---------- test 2: decoys whose price was also observed that day
    log("")
    log("2. negatives restricted to items whose price was ALSO observed that day")
    obs_by_day = {}
    for t_ in np.unique(day):
        obs_by_day[int(t_)] = np.flatnonzero(lines[:, int(t_)] > 0)
    neg2 = np.zeros((B, a.n_neg), dtype=np.int64)
    for r_ in range(B):
        pool = obs_by_day[int(day[r_])]
        pool = pool[pool != item[r_]]
        if len(pool) < a.n_neg:
            neg2[r_] = rng.choice(d.J, size=a.n_neg, p=d.neg_p)
        else:
            w = d.neg_p[pool]; w = w / w.sum()
            neg2[r_] = rng.choice(pool, size=a.n_neg, replace=False, p=w)
    built2 = build(neg2)
    allm = np.ones(B, dtype=bool)
    o2 = score_all(*built2, allm, "matched")
    res["matched_decoys"] = o2
    o1 = score_all(*built, allm, "ordinary")
    res["ordinary_decoys"] = o1
    for lb in o1:
        log(f"   {lb:18s} ordinary {o1[lb]:+.4f}   matched {o2[lb]:+.4f}   "
            f"change {o2[lb] - o1[lb]:+.4f}")
    m0 = a.labels[0]
    log(f"   -> margin over the baseline: ordinary "
        f"{o1[m0] - o1['household repeat']:+.3f}, matched "
        f"{o2[m0] - o2['household repeat']:+.3f}")

    with open(os.path.join(OUT, "price_leak_test.json"), "w") as f:
        json.dump(res, f, indent=2)
    log("")
    log("wrote out/price_leak_test.json")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--labels", nargs="+", default=["nested", "nested_noctx"])
    p.add_argument("--n-baskets", type=int, default=4000)
    p.add_argument("--n-neg", type=int, default=20)
    p.add_argument("--w-repeat", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    main(p.parse_args())
