"""
Stage 44 -- Learned baselines, scored on the SAME exact conditional as our model.

36_strong_baselines.py fits HPF and B-Emb but scores everything on the old 21-way sampled
metric, which the project has abandoned: its optimal scorer is log p - log q, not log p,
so it flatters whichever model absorbed the proposal.  Those numbers are not comparable to
anything the code now computes.

This refits both baselines and scores them exactly where our model is scored: a softmax
over the purchased product's category, minus whatever else the basket holds from that
category (Eq. 18 of the specification).  Median 47 admissible products, so chance is
-log 47 = -3.85.  Every model sees the identical choice set, so the comparison is
like-for-like on the metric, though not on tuning budget -- see the note at the end.

  HPF     hierarchical Poisson factorisation (Gopalan et al.).  Non-negative
          household x product factors fitted to purchase counts.  Has the household half
          of our model and none of the basket half.
  B-Emb   Bernoulli / exponential-family embeddings (Rudolph et al.).  Skip-gram with
          negative sampling over within-basket product pairs.  Has the basket half and
          none of the household half.

Writes out/baselines_exact.json.
"""
import argparse
import importlib
import json
import os
import sys

import numpy as np
import torch

# the model lives in ../model; add it to the path so `27_nested_basket` and
# `28_nested_counterfactual` resolve by their bare module names.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "model"))
nb = importlib.import_module("27_nested_basket")
cf = importlib.import_module("28_nested_counterfactual")

HERE = os.path.dirname(os.path.abspath(__file__))
IN = os.path.join(HERE, "..", "..", "basket_input")
OUT = os.path.join(HERE, "..", "..", "out")


def log(m):
    print(f"[44] {m}", flush=True)


def fit_hpf(d, K, iters, lr, seed, dev):
    """y_ui ~ Poisson(theta_u . beta_i) with non-negative factors."""
    tr = d.splits["train"]
    y = np.zeros((d.N, d.J), dtype=np.float32)
    np.add.at(y, (tr["user"], tr["item"]), 1.0)
    obs = np.flatnonzero(y.sum(1) > 0)
    g = torch.Generator().manual_seed(seed)
    th = torch.nn.Parameter(torch.rand(d.N, K, generator=g) * .1 + .05)
    be = torch.nn.Parameter(torch.rand(d.J, K, generator=g) * .1 + .05)
    opt = torch.optim.Adam([th, be], lr=lr)
    Y = torch.as_tensor(y)
    rng = np.random.default_rng(seed)
    log(f"  HPF: {int((y > 0).sum()):,} non-zero (household, product) cells, K={K}")
    for it in range(iters):
        u = torch.as_tensor(rng.choice(obs, size=256))
        lam = (th[u].clamp_min(1e-6) @ be.clamp_min(1e-6).T).clamp_min(1e-8)
        loss = (lam - Y[u] * torch.log(lam)).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            th.clamp_(min=1e-6); be.clamp_(min=1e-6)
    return th.detach(), be.detach()


def fit_bemb(d, K, iters, lr, seed, dev, n_neg=5):
    """Skip-gram with negative sampling over within-basket product pairs."""
    tr = d.splits["train"]
    P = []
    for i in range(tr["n_baskets"]):
        it = tr["item"][tr["starts"][i]:tr["ends"][i]]
        if len(it) > 1:
            for a_ in it:
                for b_ in it:
                    if a_ != b_:
                        P.append((a_, b_))
    P = np.array(P, dtype=np.int64)
    log(f"  B-Emb: {len(P):,} within-basket ordered product pairs, K={K}")
    g = torch.Generator().manual_seed(seed)
    al = torch.nn.Parameter(torch.randn(d.J, K, generator=g) * .05)
    rh = torch.nn.Parameter(torch.randn(d.J, K, generator=g) * .05)
    opt = torch.optim.Adam([al, rh], lr=lr)
    rng = np.random.default_rng(seed)
    for it in range(iters):
        b = P[rng.integers(0, len(P), size=8192)]
        neg = torch.as_tensor(rng.choice(d.J, size=(8192, n_neg), p=d.neg_p))
        c, t = torch.as_tensor(b[:, 0]), torch.as_tensor(b[:, 1])
        pos = (rh[c] * al[t]).sum(-1)
        ng = (rh[c].unsqueeze(1) * al[neg]).sum(-1)
        loss = -(torch.nn.functional.logsigmoid(pos).mean()
                 + torch.nn.functional.logsigmoid(-ng).mean())
        opt.zero_grad(); loss.backward(); opt.step()
    return al.detach(), rh.detach()


def main(a):
    dev = torch.device("cpu")
    d = nb.NestedData(IN, device=dev)
    sp = d.splits["test"]
    rng = np.random.default_rng(a.seed)

    bidx = rng.choice(sp["n_baskets"], size=min(a.n_baskets, sp["n_baskets"]),
                      replace=False)
    vidx = np.random.default_rng(a.seed + 1).choice(
        d.splits["validation"]["n_baskets"], size=1500, replace=False)

    def blocks(split, bi):
        s = d.splits[split]
        rows = np.concatenate([np.arange(s["starts"][i], s["ends"][i]) for i in bi])
        user, item = s["user"][rows], s["item"][rows]
        cats = d.item_cat_np[item]
        cand = d.cat_items_np[cats]
        M = cand.shape[1]
        tgt = torch.as_tensor(d.item_pos_np[item])
        avail = torch.as_tensor(d.cat_mask_np[cats] > 0)
        avail &= d.carried[torch.as_tensor(cand),
                           torch.as_tensor(s["store"][rows]).unsqueeze(1)]
        own = np.concatenate([[k] * (s["ends"][i] - s["starts"][i])
                              for k, i in enumerate(bi)])
        key_pos = own.astype(np.int64) * d.J + item
        key_c = own[:, None].astype(np.int64) * d.J + cand
        avail &= torch.as_tensor(~np.isin(key_c, key_pos))
        ar = torch.arange(len(rows))
        avail[ar, tgt] = True
        return user, item, cand, tgt, avail, ar, rows

    vu, vi, vc, vt, va, var, _ = blocks("validation", vidx)
    tu, ti, tc, tt, ta, tar, trows = blocks("test", bidx)
    log(f"{len(tt):,} held-out purchases; "
        f"{float(ta.float().sum(1).mean()):.1f} admissible products per row "
        f"(chance {-np.log(float(ta.float().sum(1).mean())):.4f})")

    def score(s_test, s_val, name, grid):
        best = max(((float(torch.log_softmax(
            s_val.masked_fill(~va, -1e9) * w, 1)[var, vt].mean()), w) for w in grid))
        w = best[1]
        s = (s_test * w).masked_fill(~ta, -1e9)
        lp = torch.log_softmax(s, 1)[tar, tt]
        top1 = float((s.argmax(1) == tt).float().mean())
        log(f"  {name:22s} w={w:<5g} loglik {float(lp.mean()):+.4f}  top-1 {top1:.3f}")
        return float(lp.mean()), top1, lp.numpy(), w

    res = {"n_rows": int(len(tt)),
           "admissible": float(ta.float().sum(1).mean()), "models": {}}
    per = {}

    log("")
    log("fitting HPF ...")
    th, be = fit_hpf(d, a.K, a.iters, a.lr, a.seed, dev)
    hs = lambda U, C: torch.log((th[torch.as_tensor(U)].unsqueeze(1)
                                 * be[torch.as_tensor(C)]).sum(-1).clamp_min(1e-8))
    r = score(hs(tu, tc), hs(vu, vc), "HPF", a.grid)
    res["models"]["HPF"], per["HPF"] = {"loglik": r[0], "top1": r[1], "w": r[3]}, r[2]

    log("")
    log("fitting B-Emb ...")
    al, rh = fit_bemb(d, a.K, a.iters, a.lr, a.seed, dev)

    def bs(split, bi, U, C, ar_):
        s = d.splits[split]
        rows = np.concatenate([np.arange(s["starts"][i], s["ends"][i]) for i in bi])
        own = np.concatenate([[k] * (s["ends"][i] - s["starts"][i])
                              for k, i in enumerate(bi)])
        ctx = torch.zeros(len(rows), al.shape[1])
        for k in np.unique(own):
            q = np.flatnonzero(own == k)
            if len(q) > 1:
                tot = rh[torch.as_tensor(s["item"][rows[q]])].sum(0)
                for z, rr in enumerate(q):
                    ctx[rr] = (tot - rh[int(s["item"][rows[q[z]]])]) / (len(q) - 1)
        return (al[torch.as_tensor(C)] * ctx.unsqueeze(1)).sum(-1)
    r = score(bs("test", bidx, tu, tc, tar), bs("validation", vidx, vu, vc, var),
              "B-Emb", a.grid)
    res["models"]["B-Emb"], per["B-Emb"] = {"loglik": r[0], "top1": r[1], "w": r[3]}, r[2]

    log("")
    log("our model, identical choice sets ...")
    for lab in a.labels:
        if not os.path.exists(os.path.join(OUT, f"{lab}_nested.pt")):
            continue
        m, _ = cf.load(lab, d, dev)
        bt = nb.make_batch(d, m, "test", bidx, np.random.default_rng(a.seed), dev)
        with torch.no_grad():
            u = m.item_utility(bt["user"], bt["cand"], bt["ctx"], bt["dlogp"],
                               bt["state"], bt["week"], bt["store"])
        u = u.masked_fill(~bt["avail"], -1e9)
        ar = torch.arange(u.shape[0])
        lp = torch.log_softmax(u, 1)[ar, bt["target"]]
        t1 = float((u.argmax(1) == bt["target"]).float().mean())
        log(f"  {lab:22s} {'':10s} loglik {float(lp.mean()):+.4f}  top-1 {t1:.3f}")
        res["models"][lab] = {"loglik": float(lp.mean()), "top1": t1}
        per[lab] = lp.numpy()

    # paired household bootstrap on the gap to the best baseline
    hh = sp["user"][trows]
    uh = np.unique(hh)
    idx = {h: np.flatnonzero(hh == h) for h in uh}
    r2 = np.random.default_rng(a.seed + 2)
    draws = [np.concatenate([idx[h] for h in r2.choice(uh, len(uh), True)])
             for _ in range(a.reps)]
    best_base = max(("HPF", "B-Emb"), key=lambda k: res["models"][k]["loglik"])
    ours = a.labels[0]
    if ours in per:
        gd = np.array([per[ours][ix].mean() - per[best_base][ix].mean() for ix in draws])
        lo, hi = np.percentile(gd, [2.5, 97.5])
        res["gap_vs_best_baseline"] = {"baseline": best_base,
                                       "gap": float(gd.mean()),
                                       "ci": [float(lo), float(hi)]}
        log("")
        log(f"  {ours} minus {best_base}: {gd.mean():+.4f} nats "
            f"[{lo:+.4f}, {hi:+.4f}]  (paired household bootstrap)")

    with open(os.path.join(OUT, "baselines_exact.json"), "w") as f:
        json.dump(res, f, indent=2)
    log("")
    log("NOTE: same metric and same choice sets, but NOT the same tuning budget -- each")
    log("      baseline has one tuned scalar, our model has 870k parameters and an")
    log("      architecture chosen over many runs.  Read it as a floor, not a fair fight.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--labels", nargs="+", default=["ps_nested", "ps_off"])
    p.add_argument("--K", type=int, default=64)
    p.add_argument("--iters", type=int, default=4000)
    p.add_argument("--lr", type=float, default=0.02)
    p.add_argument("--grid", type=float, nargs="+",
                   default=[0.1, 0.25, 0.5, 1, 2, 4, 8])
    p.add_argument("--n-baskets", type=int, default=4000)
    p.add_argument("--reps", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    main(p.parse_args())
