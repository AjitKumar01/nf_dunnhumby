"""
Stage 36 -- The baselines SHOPPER compares against, and its price-variation evaluation.

NESTED_MODEL.md 8.1c compares against random, popularity, household repeat-purchase and
raw item-item co-occurrence.  Three of those are weak by construction and the fourth --
raw co-occurrence -- scores BELOW random, which flatters our model: it is a counting
heuristic standing in for what should be a *learned* item-item model.

SHOPPER (Ruiz, Athey & Blei 2020, Table 2) compares against two learned latent-factor
models, and they are the right competitors:

  HPF     hierarchical Poisson factorization (Gopalan et al. 2015).  Factorises the
          user x item matrix.  Captures user preferences, no item-item interaction, no
          price.  Here: Poisson matrix factorisation fitted by SGD, which is the same
          likelihood without the hierarchical Gamma priors.
  B-Emb   Bernoulli / exponential-family embeddings (Rudolph et al. 2016).  Captures
          item-item interaction only, no user preference, no price.  Here: skip-gram
          with negative sampling over within-basket pairs, which is the same objective.

Together they bracket our model: HPF has the user half, B-Emb has the basket half, and
neither has price.  If we do not beat both, the combination is not earning its keep.

SHOPPER also runs a second evaluation we have never run: score on held-out baskets whose
prices deviate MOST from their item's average.  That is where a counterfactual price
model has to prove itself, and it is a much harder subset than the average trip.  We
report every model on the full test set and on the top price-variation decile.

Writes out/strong_baselines.json.
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
IN = os.path.join(HERE, "..", "..", "basket_input")
OUT = os.path.join(HERE, "..", "..", "out")


def log(m):
    print(f"[36] {m}", flush=True)


def fit_poisson_mf(d, K, iters, lr, seed, dev):
    """HPF analogue: y_ui ~ Poisson(theta_u . beta_i), non-negative factors.

    Fitted on the training user x item count matrix by SGD over observed cells plus an
    equal number of sampled zeros, which is the standard implicit-feedback treatment.
    """
    tr = d.splits["train"]
    cnt = {}
    for u, i in zip(tr["user"], tr["item"]):
        cnt[(u, i)] = cnt.get((u, i), 0) + 1
    uu = np.array([k[0] for k in cnt], dtype=np.int64)
    ii = np.array([k[1] for k in cnt], dtype=np.int64)
    yy = np.array(list(cnt.values()), dtype=np.float32)
    log(f"  HPF: {len(yy):,} observed (household, item) cells")
    g = torch.Generator().manual_seed(seed)
    th = torch.nn.Parameter(torch.randn(d.N, K, generator=g) * 0.1 - 1.0)
    be = torch.nn.Parameter(torch.randn(d.J, K, generator=g) * 0.1 - 1.0)
    opt = torch.optim.Adam([th, be], lr=lr)
    rng = np.random.default_rng(seed)
    ut, it_, yt = (torch.as_tensor(uu, device=dev), torch.as_tensor(ii, device=dev),
                   torch.as_tensor(yy, device=dev))
    for t in range(iters):
        idx = torch.as_tensor(rng.integers(0, len(yy), size=8192), device=dev)
        u_, i_, y_ = ut[idx], it_[idx], yt[idx]
        zu = torch.as_tensor(rng.integers(0, d.N, size=8192), device=dev)
        zi = torch.as_tensor(rng.integers(0, d.J, size=8192), device=dev)
        rate_p = (torch.nn.functional.softplus(th[u_]) *
                  torch.nn.functional.softplus(be[i_])).sum(-1).clamp(1e-6, 50)
        rate_z = (torch.nn.functional.softplus(th[zu]) *
                  torch.nn.functional.softplus(be[zi])).sum(-1).clamp(1e-6, 50)
        loss = (rate_p - y_ * torch.log(rate_p)).mean() + rate_z.mean()
        loss = loss + 1e-4 * (th ** 2).sum() / 8192 + 1e-4 * (be ** 2).sum() / 8192
        opt.zero_grad(); loss.backward(); opt.step()
    return (torch.nn.functional.softplus(th.detach()),
            torch.nn.functional.softplus(be.detach()))


def fit_item_embeddings(d, K, iters, lr, seed, dev, n_neg=5):
    """B-Emb analogue: skip-gram with negative sampling over within-basket item pairs.

    Maximises log sigmoid(v_j . c_k) for items co-occurring in a basket and
    log sigmoid(-v_j . c_n) for sampled non-neighbours -- the same objective exponential
    family (Bernoulli) embeddings optimise, with the same zero-weighting role played by
    the negative sample count.
    """
    tr = d.splits["train"]
    pairs = []
    for s, e in zip(tr["starts"], tr["ends"]):
        u = np.unique(tr["item"][s:e])
        if len(u) < 2 or len(u) > 40:
            continue
        for x in range(len(u)):
            for y in range(len(u)):
                if x != y:
                    pairs.append((u[x], u[y]))
    P = np.asarray(pairs, dtype=np.int64)
    log(f"  B-Emb: {len(P):,} within-basket ordered item pairs")
    g = torch.Generator().manual_seed(seed)
    V = torch.nn.Parameter(torch.randn(d.J, K, generator=g) * 0.05)
    Cc = torch.nn.Parameter(torch.randn(d.J, K, generator=g) * 0.05)
    opt = torch.optim.Adam([V, Cc], lr=lr)
    rng = np.random.default_rng(seed)
    Pt = torch.as_tensor(P, device=dev)
    for t in range(iters):
        idx = torch.as_tensor(rng.integers(0, len(P), size=8192), device=dev)
        j, k = Pt[idx, 0], Pt[idx, 1]
        neg = torch.as_tensor(rng.choice(d.J, size=(8192, n_neg), p=d.neg_p),
                              device=dev)
        pos = torch.nn.functional.logsigmoid((V[j] * Cc[k]).sum(-1))
        ng = torch.nn.functional.logsigmoid(-(V[j].unsqueeze(1) * Cc[neg]).sum(-1))
        loss = -(pos.mean() + ng.sum(-1).mean())
        loss = loss + 1e-4 * ((V ** 2).sum() + (Cc ** 2).sum()) / 8192
        opt.zero_grad(); loss.backward(); opt.step()
    return V.detach(), Cc.detach()


def main(a):
    dev = torch.device(a.device)
    d = nb.NestedData(IN, device=dev)
    sp = d.splits["test"]
    rng = np.random.default_rng(a.seed)

    bidx = rng.choice(sp["n_baskets"], size=min(a.n_baskets, sp["n_baskets"]),
                      replace=False)
    rows, owner, bitems = [], [], []
    for bi, i in enumerate(bidx):
        r = np.arange(sp["starts"][i], sp["ends"][i])
        rows.extend(r.tolist()); owner.extend([bi] * len(r))
        bitems.append(sp["item"][r])
    rows = np.asarray(rows); owner = np.asarray(owner)
    user, item = sp["user"][rows], sp["item"][rows]
    day, week = sp["day"][rows], sp["week"][rows]
    store, rw = sp["store"][rows], sp["raw_week"][rows]
    B = len(rows)

    neg = rng.choice(d.J, size=(B, a.n_neg), p=d.neg_p).astype(np.int64)
    cnp = np.concatenate([item[:, None], neg], axis=1); M = cnp.shape[1]
    cand = torch.as_tensor(cnp, device=dev)
    dr = np.repeat(day[:, None], M, 1); ur = np.repeat(user[:, None], M, 1)
    sr = np.repeat(store[:, None], M, 1); rr = np.repeat(rw[:, None], M, 1)

    # price-variation decile: how far the TRUE item's price sits from its own mean
    dev_true = np.abs(d.log_price_dev[torch.as_tensor(item, device=dev),
                                      torch.as_tensor(day, device=dev)].cpu().numpy())
    hi = dev_true >= np.quantile(dev_true, 1 - a.top_frac)
    log(f"{B:,} held-out rows, {a.n_neg} negatives; "
        f"{hi.sum():,} in the top {a.top_frac:.0%} by |price deviation| "
        f"(|dev| >= {np.quantile(dev_true, 1 - a.top_frac):.3f})")

    def report(name, s):
        s = s.masked_fill(~availab, -1e9)
        lp = torch.log_softmax(s, dim=1)[:, 0].cpu().numpy()
        t1 = (s.argmax(1) == 0).cpu().numpy()
        return {"loglik": float(lp.mean()), "top1": float(t1.mean()),
                "loglik_high_price": float(lp[hi].mean()),
                "top1_high_price": float(t1[hi].mean())}

    # ---- a VALIDATION set, used only to tune each baseline's temperature.
    # Scoring an unscaled dot product against a fitted model is not a fair contest: the
    # softmax is sensitive to the scale of its inputs, and our model had its
    # hyperparameters selected.  Each baseline gets the same courtesy.
    vsp = d.splits["validation"]
    vrng = np.random.default_rng(a.seed + 1)
    vb = vrng.choice(vsp["n_baskets"], size=min(1500, vsp["n_baskets"]), replace=False)
    vrows, vowner, vbitems = [], [], []
    for bi, i in enumerate(vb):
        r = np.arange(vsp["starts"][i], vsp["ends"][i])
        vrows.extend(r.tolist()); vowner.extend([bi] * len(r))
        vbitems.append(vsp["item"][r])
    vrows = np.asarray(vrows); vowner = np.asarray(vowner)
    vuser, vitem = vsp["user"][vrows], vsp["item"][vrows]
    vstore = vsp["store"][vrows]
    VB = len(vrows)
    vneg = vrng.choice(d.J, size=(VB, a.n_neg), p=d.neg_p).astype(np.int64)
    vcnp = np.concatenate([vitem[:, None], vneg], axis=1)
    vcand = torch.as_tensor(vcnp, device=dev)
    vsr = np.repeat(vstore[:, None], vcnp.shape[1], 1)
    vavail = d.carried[vcand, torch.as_tensor(vsr, device=dev)].clone()
    vavail[:, 0] = True

    def tune(score_fn, grid, name):
        best = (-1e9, None)
        for w in grid:
            s = score_fn(w).masked_fill(~vavail, -1e9)
            ll = float(torch.log_softmax(s, dim=1)[:, 0].mean())
            if ll > best[0]:
                best = (ll, w)
        log(f"  {name}: temperature {best[1]} (validation loglik {best[0]:.4f})")
        return best[1]

    res, order = {}, []

    # availability mask shared by every model, so the choice sets are identical
    availab = d.carried[cand, torch.as_tensor(sr, device=dev)].clone()
    availab[:, 0] = True

    log("")
    log("fitting HPF (user x item Poisson factorisation) ...")
    th, be = fit_poisson_mf(d, a.K, a.iters, a.lr, a.seed, dev)
    hpf_raw = lambda U, C: torch.log((th[torch.as_tensor(U, device=dev)].unsqueeze(1) *
                                      be[C]).sum(-1).clamp_min(1e-8))
    w = tune(lambda w: hpf_raw(vuser, vcand) * w, a.grid, "HPF")
    s = hpf_raw(user, cand) * w
    res["HPF (user x item)"] = report("HPF", s); order.append("HPF (user x item)")

    log("fitting B-Emb (item-item skip-gram) ...")
    V, Cc = fit_item_embeddings(d, a.K, a.iters, a.lr, a.seed, dev)
    ctxvec = torch.zeros(B, a.K, device=dev)
    for r_ in range(B):
        o = bitems[owner[r_]]; o = o[o != item[r_]]
        if len(o):
            ctxvec[r_] = Cc[torch.as_tensor(o, device=dev)].mean(0)
    vctx = torch.zeros(VB, a.K, device=dev)
    for r_ in range(VB):
        o = vbitems[vowner[r_]]; o = o[o != vitem[r_]]
        if len(o):
            vctx[r_] = Cc[torch.as_tensor(o, device=dev)].mean(0)
    w = tune(lambda w: (V[vcand] * vctx.unsqueeze(1)).sum(-1) * w, a.grid, "B-Emb")
    s = (V[cand] * ctxvec.unsqueeze(1)).sum(-1) * w
    res["B-Emb (item-item)"] = report("B-Emb", s); order.append("B-Emb (item-item)")

    log("scoring the nested model ...")
    for lb in a.labels:
        if not os.path.exists(os.path.join(OUT, f"{lb}_nested.pt")):
            continue
        m, _ = cf.load(lb, d, dev)
        st = torch.as_tensor(d.state(ur.ravel(), cnp.ravel(), dr.ravel()).reshape(
            B, M, nb.N_STATE_FEATURES), device=dev)
        dl = d.log_price_dev[cand, torch.as_tensor(dr, device=dev)]
        if m.use_store and m.use_store_price:
            dl = dl + d.store_dev(cnp.ravel(), sr.ravel(), rr.ravel()).reshape(B, M)
        A = m.alpha.detach()
        ctx = torch.zeros(B, m.K, device=dev)
        for r_ in range(B):
            o = bitems[owner[r_]]; o = o[o != item[r_]]
            if len(o):
                ctx[r_] = A[torch.as_tensor(o, device=dev)].mean(0)
        with torch.no_grad():
            s = m.item_utility(torch.as_tensor(user, device=dev), cand, ctx, dl, st,
                               torch.as_tensor(week, device=dev),
                               torch.as_tensor(store, device=dev))
        res[f"nested ({lb})"] = report(lb, s); order.append(f"nested ({lb})")

    log("")
    log(f"{'model':26s}{'loglik':>10s}{'top-1':>8s}   |  {'loglik':>10s}{'top-1':>8s}  (top "
        f"{a.top_frac:.0%} price variation)")
    for k in order:
        v = res[k]
        log(f"{k:26s}{v['loglik']:10.4f}{v['top1']:8.3f}   |  "
            f"{v['loglik_high_price']:10.4f}{v['top1_high_price']:8.3f}")

    res["_meta"] = {"rows": int(B), "n_neg": a.n_neg, "K": a.K,
                    "high_price_rows": int(hi.sum()), "top_frac": a.top_frac}
    with open(os.path.join(OUT, "strong_baselines.json"), "w") as f:
        json.dump(res, f, indent=2)
    log("")
    log("wrote out/strong_baselines.json")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--labels", nargs="+", default=["nested", "nested_both"])
    p.add_argument("--n-baskets", type=int, default=4000)
    p.add_argument("--n-neg", type=int, default=20)
    p.add_argument("--K", type=int, default=50)
    p.add_argument("--iters", type=int, default=4000)
    p.add_argument("--lr", type=float, default=0.02)
    p.add_argument("--grid", type=float, nargs="+",
                   default=[0.25, 0.5, 1, 2, 4, 8, 16],
                   help="temperatures tried on VALIDATION for each baseline")
    p.add_argument("--top-frac", type=float, default=0.10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    main(p.parse_args())
