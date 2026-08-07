"""
Stage 38 -- The item head scored WITHOUT any negative sampling.

Every item likelihood in this repository is a 21-way discrimination: the true item plus
20 decoys drawn from q(m) prop count(m)^0.75.  That number has a hidden dependence on q.
The Bayes-optimal scorer for it is not log p(m) but log p(m) - log q(m), because a popular
decoy is more likely to appear than a rare one and the scorer should discount it.  So a
model that learns the TRUE choice probability is at a disadvantage on that metric, and a
model that has silently absorbed -log q into its intercept is flattered by it.

The catalogue here has 5,455 items, which is small enough to just do the sum exactly:

    log P(j | household, basket, prices, store) = u_j - log sum_{m stocked} exp(u_m)

No proposal, no correction, nothing to tune.  This is the quantity the generator actually
samples from, and the only item metric in this repository that is not a function of how
the decoys were drawn.

Reported per checkpoint directory so a pre-fix and post-fix model can be compared:

    exact     full-catalogue log-likelihood, top-1, top-10, MRR, over stocked items
    pop-21    the historical 21-way number, popularity decoys
    unif-21   the same but decoys uniform over the store's stocked items, so log q is
              constant and cancels -- a sampled metric with no proposal tilt

Writes out/exact_likelihood.json.
"""
import argparse
import importlib
import json
import os

import numpy as np
import torch

nb = importlib.import_module("27_nested_basket")

HERE = os.path.dirname(os.path.abspath(__file__))
IN = os.path.join(HERE, "..", "basket_input")
OUT = os.path.join(HERE, "..", "out")


def log(m):
    print(f"[38] {m}", flush=True)


def build_model(d, cfg, dev):
    m = nb.NestedModel(
        d, K=cfg["K"], Kp=cfg["Kp"], Kt=cfg["Kt"], Ks=cfg["Ks"], seed=cfg["seed"],
        use_nest=not cfg["no_nest"], use_quantity=not cfg["no_quantity"],
        use_store=not cfg["no_store"],
        use_store_price=not (cfg["no_store_price"] or cfg["avail_only"]),
        avail_only=cfg["avail_only"], use_state=not cfg["no_state"],
        use_breadth=not cfg["no_breadth"], use_context=not cfg["no_context"],
        ctx_agg=cfg.get("ctx_agg", "mean"),
        learn_ctx_scale=cfg.get("learn_ctx_scale", False),
        use_cat_context=cfg.get("cat_context", False),
        use_cat_pair=cfg.get("cat_pair", False),
        untie_rho=cfg.get("untie_rho", False),
        prefix_context=cfg.get("prefix_context", False),
        neg_in_cat=cfg.get("neg_in_cat", 0.0),
        item_loss=cfg.get("item_loss", "softmax")).to(dev)
    return m


@torch.no_grad()
def exact(m, d, dev, rows_data, chunk=256):
    """Full-catalogue log P(true item).  Sums over every item the store stocks."""
    user, item, day, week, store, rw, ctx = rows_data
    B = len(item)
    J = d.J
    allj = torch.arange(J, device=dev)
    tot = 0.0
    hit1 = hit10 = 0
    rr = 0.0
    for s in range(0, B, chunk):
        e = min(s + chunk, B)
        n = e - s
        cand = allj.unsqueeze(0).expand(n, J)
        day_r = np.repeat(day[s:e, None], J, 1)
        user_r = np.repeat(user[s:e, None], J, 1)
        store_r = np.repeat(store[s:e, None], J, 1)
        rw_r = np.repeat(rw[s:e, None], J, 1)
        st = torch.as_tensor(
            d.state(user_r.ravel(), cand.cpu().numpy().ravel(), day_r.ravel()).reshape(
                n, J, nb.N_STATE_FEATURES), device=dev)
        dl = d.log_price_dev[cand, torch.as_tensor(day_r, device=dev)]
        if m.use_store and m.use_store_price:
            dl = dl + d.store_dev(cand.cpu().numpy().ravel(), store_r.ravel(),
                                  rw_r.ravel()).reshape(n, J)
        u = m.item_utility(torch.as_tensor(user[s:e], device=dev), cand, ctx[s:e], dl, st,
                           torch.as_tensor(week[s:e], device=dev),
                           torch.as_tensor(store[s:e], device=dev))
        av = torch.ones(n, J, dtype=torch.bool, device=dev)
        if m.use_store:
            av = d.carried[cand, torch.as_tensor(store_r, device=dev)].clone()
        tru = torch.as_tensor(item[s:e], device=dev)
        av[torch.arange(n, device=dev), tru] = True   # the bought item was available
        u = u.masked_fill(~av, -1e9)
        lp = torch.log_softmax(u, dim=1)
        tgt = lp[torch.arange(n, device=dev), tru]
        tot += float(tgt.sum())
        rank = (u > u[torch.arange(n, device=dev), tru].unsqueeze(1)).sum(1) + 1
        hit1 += int((rank == 1).sum()); hit10 += int((rank <= 10).sum())
        rr += float((1.0 / rank.float()).sum())
    return tot / B, hit1 / B, hit10 / B, rr / B


@torch.no_grad()
def sampled(m, d, dev, rows_data, neg, chunk=512):
    user, item, day, week, store, rw, ctx = rows_data
    B = len(item)
    M = neg.shape[1] + 1
    cnp = np.concatenate([item[:, None], neg], axis=1)
    tot = 0.0; hit = 0
    for s in range(0, B, chunk):
        e = min(s + chunk, B); n = e - s
        c = cnp[s:e]
        cand = torch.as_tensor(c, device=dev)
        day_r = np.repeat(day[s:e, None], M, 1)
        user_r = np.repeat(user[s:e, None], M, 1)
        store_r = np.repeat(store[s:e, None], M, 1)
        rw_r = np.repeat(rw[s:e, None], M, 1)
        st = torch.as_tensor(
            d.state(user_r.ravel(), c.ravel(), day_r.ravel()).reshape(
                n, M, nb.N_STATE_FEATURES), device=dev)
        dl = d.log_price_dev[cand, torch.as_tensor(day_r, device=dev)]
        if m.use_store and m.use_store_price:
            dl = dl + d.store_dev(c.ravel(), store_r.ravel(), rw_r.ravel()).reshape(n, M)
        u = m.item_utility(torch.as_tensor(user[s:e], device=dev), cand, ctx[s:e], dl, st,
                           torch.as_tensor(week[s:e], device=dev),
                           torch.as_tensor(store[s:e], device=dev))
        av = torch.ones(n, M, dtype=torch.bool, device=dev)
        if m.use_store:
            av = d.carried[cand, torch.as_tensor(store_r, device=dev)].clone()
        av[:, 0] = True
        u = u.masked_fill(~av, -1e9)
        lp = torch.log_softmax(u, dim=1)[:, 0]
        tot += float(lp.sum()); hit += int((u.argmax(1) == 0).sum())
    return tot / B, hit / B


def main(a):
    dev = torch.device(a.device)
    d = nb.NestedData(IN, device=dev)
    sp = d.splits[a.split]
    rng = np.random.default_rng(a.seed)

    bidx = rng.choice(sp["n_baskets"], size=min(a.n_baskets, sp["n_baskets"]),
                      replace=False)
    rows, owner = [], []
    for bi, i in enumerate(bidx):
        r = np.arange(sp["starts"][i], sp["ends"][i])
        rows.extend(r.tolist()); owner.extend([bi] * len(r))
    rows = np.asarray(rows); owner = np.asarray(owner)
    user, item = sp["user"][rows], sp["item"][rows]
    day, week = sp["day"][rows], sp["week"][rows]
    store = sp["store"][rows]
    rw = sp["raw_week"][rows]
    B = len(rows)
    log(f"{B:,} held-out purchase rows in {len(bidx):,} baskets, catalogue {d.J:,} items")

    # the same decoys for every model, drawn once
    neg_pop = rng.choice(d.J, size=(B, a.n_neg), p=d.neg_p).astype(np.int64)
    neg_unif = rng.integers(0, d.J, size=(B, a.n_neg)).astype(np.int64)

    res = {"rows": int(B), "baskets": int(len(bidx)), "split": a.split,
           "catalogue": int(d.J), "models": {}}

    for tag, ckdir in [("before", a.before), ("after", a.after)]:
        if ckdir is None or not os.path.isdir(ckdir):
            continue
        for lb in a.labels:
            ck = os.path.join(ckdir, f"{lb}_nested.pt")
            hj = os.path.join(OUT, f"{lb}_nested_history.json")
            if not os.path.exists(ck) or not os.path.exists(hj):
                log(f"  {tag}/{lb}: missing, skipping")
                continue
            cfg = json.load(open(hj))["config"]
            m = build_model(d, cfg, dev)
            m.load_state_dict(torch.load(ck, map_location=dev)); m.eval()

            A = m.alpha.detach()
            ctx = torch.zeros(B, m.K, device=dev)
            for bi in range(len(bidx)):
                r = np.flatnonzero(owner == bi)
                if len(r) > 1:
                    its = item[r]
                    tot = A[torch.as_tensor(its, device=dev)].sum(0)
                    for q, rr_ in enumerate(r):
                        ctx[rr_] = (tot - A[its[q]]) / (len(its) - 1)

            rd = (user, item, day, week, store, rw, ctx)
            ell, h1, h10, mrr = exact(m, d, dev, rd)
            pll, pacc = sampled(m, d, dev, rd, neg_pop)
            ull, uacc = sampled(m, d, dev, rd, neg_unif)
            res["models"][f"{tag}/{lb}"] = {
                "exact_loglik": ell, "exact_top1": h1, "exact_top10": h10, "exact_mrr": mrr,
                "pop21_loglik": pll, "pop21_top1": pacc,
                "unif21_loglik": ull, "unif21_top1": uacc}
            log(f"  {tag:6s} {lb:16s} exact {ell:8.4f} (top1 {h1:.3f} top10 {h10:.3f} "
                f"mrr {mrr:.3f})   pop-21 {pll:+.4f}   unif-21 {ull:+.4f}")

    with open(os.path.join(OUT, "exact_likelihood.json"), "w") as f:
        json.dump(res, f, indent=2)
    log("")
    log("wrote out/exact_likelihood.json")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--labels", nargs="+", default=["nested", "nested_both", "nested_hn75"])
    p.add_argument("--before", default=None, help="directory of pre-fix checkpoints")
    p.add_argument("--after", default=OUT, help="directory of post-fix checkpoints")
    p.add_argument("--split", default="test")
    p.add_argument("--n-baskets", type=int, default=1500)
    p.add_argument("--n-neg", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    main(p.parse_args())
