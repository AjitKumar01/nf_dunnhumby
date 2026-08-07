"""
Stage 35 -- What does the model do when the basket is NOT known?

Every item log-likelihood in NESTED_MODEL.md is CONDITIONAL on the rest of the basket:
scoring item j uses the leave-one-out mean of alpha over the other n-1 items.  That is a
fill-in-the-blank score.  A recommender does not have that.  At the moment a
recommendation is made the cart holds 0, 1, 2, ... items -- never all of them minus the
one being predicted.

So this script asks the question the deployment setting actually poses:

  1. TRAIN/TEST MISMATCH.  Take the model trained WITH context and score it with the
     context zeroed.  Is it worse than a model TRAINED without context?  If so, the
     context-trained model is actively harmful when deployed without a basket, and the
     reported numbers overstate what it can do.

  2. THE DEGRADATION CURVE.  Score the held-out item given a random k-item prefix of its
     basket, for k = 0, 1, 2, 4, 8, and all-but-one.  This is the curve a recommender
     lives on: it says how many items must already be in the cart before the interaction
     term earns anything.

  3. AGAINST BASELINES.  At k = 0 -- the cold-start recommendation -- does the model
     still beat household repeat-purchase, which needs no basket at all?

Everything is scored through one scorer on identical candidate sets, and the candidates
are drawn once and reused across every model and every k, so the columns are comparable.

Writes out/context_ablation.json and figures/context_ablation.png.
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
FIG = os.path.join(HERE, "..", "figures")
PAL = {"blue": "#2d6cdf", "grey": "#9aa5b1", "red": "#d1495b", "green": "#2a9d8f"}


def log(m):
    print(f"[35] {m}", flush=True)


@torch.no_grad()
def score(m, d, dev, rows, cand, avail, ctx, dlogp, state, week, store, user):
    u = m.item_utility(user, cand, ctx, dlogp, state, week, store)
    u = u.masked_fill(~avail, -1e9)
    lp = torch.log_softmax(u, dim=1)[:, 0]
    hit = (u.argmax(1) == 0)
    return float(lp.mean()), float(hit.float().mean())


def main(a):
    os.makedirs(FIG, exist_ok=True)
    dev = torch.device(a.device)
    d = nb.NestedData(IN, device=dev)
    sp = d.splits["test"]
    rng = np.random.default_rng(a.seed)

    # ---- one fixed evaluation set: rows, candidates, and the basket each row sits in
    bidx = rng.choice(sp["n_baskets"], size=min(a.n_baskets, sp["n_baskets"]),
                      replace=False)
    rows, owner, basket_items = [], [], []
    for bi, i in enumerate(bidx):
        r = np.arange(sp["starts"][i], sp["ends"][i])
        rows.extend(r.tolist())
        owner.extend([bi] * len(r))
        basket_items.append(sp["item"][r])
    rows = np.asarray(rows)
    owner = np.asarray(owner)
    user, item = sp["user"][rows], sp["item"][rows]
    day, week = sp["day"][rows], sp["week"][rows]
    store, rw = sp["store"][rows], sp["raw_week"][rows]
    B = len(rows)
    log(f"{B:,} held-out purchase rows in {len(bidx):,} baskets, "
        f"{a.n_neg} negatives each, drawn once and reused")

    neg = rng.choice(d.J, size=(B, a.n_neg), p=d.neg_p).astype(np.int64)
    cand_np = np.concatenate([item[:, None], neg], axis=1)
    M = cand_np.shape[1]
    day_r = np.repeat(day[:, None], M, 1)
    user_r = np.repeat(user[:, None], M, 1)
    store_r = np.repeat(store[:, None], M, 1)
    rw_r = np.repeat(rw[:, None], M, 1)
    cand = torch.as_tensor(cand_np, device=dev)
    st = torch.as_tensor(
        d.state(user_r.ravel(), cand_np.ravel(), day_r.ravel()).reshape(
            B, M, nb.N_STATE_FEATURES), device=dev)
    ut = torch.as_tensor(user, device=dev)
    wt = torch.as_tensor(week, device=dev)
    stt = torch.as_tensor(store, device=dev)

    res = {"rows": int(B), "baskets": int(len(bidx)), "n_neg": a.n_neg, "models": {}}

    for label in a.labels:
        ck = os.path.join(OUT, f"{label}_nested.pt")
        if not os.path.exists(ck):
            log(f"  {label}: no checkpoint, skipping")
            continue
        m, cfg = cf.load(label, d, dev)
        dlogp = d.log_price_dev[cand, torch.as_tensor(day_r, device=dev)]
        if m.use_store and m.use_store_price:
            dlogp = dlogp + d.store_dev(cand_np.ravel(), store_r.ravel(),
                                        rw_r.ravel()).reshape(B, M)
        avail = torch.ones(B, M, dtype=torch.bool, device=dev)
        if m.use_store:
            avail = d.carried[cand, torch.as_tensor(store_r, device=dev)].clone()
            avail[:, 0] = True
        A = m.alpha.detach()

        # The context the model was TRAINED against: a leave-one-out mean over the whole
        # basket.  Averaging shrinks a vector, so this norm is the reference a partial
        # cart has to match.
        full = torch.zeros(B, m.K, device=dev)
        for r_ in range(B):
            others = basket_items[owner[r_]]
            others = others[others != item[r_]]
            if len(others):
                full[r_] = A[torch.as_tensor(others, device=dev)].mean(0)
        target = float(full.norm(dim=1)[full.norm(dim=1) > 0].mean())
        scale = float(m.ctx_scale) if getattr(m, "ctx_scale", None) is not None else 1.0

        ctxs = {}
        for k in a.prefix:
            c = torch.zeros(B, m.K, device=dev)
            if k > 0:
                for r_ in range(B):
                    others = basket_items[owner[r_]]
                    others = others[others != item[r_]]
                    if len(others) == 0:
                        continue
                    pick = rng.choice(len(others), size=min(k, len(others)),
                                      replace=False)
                    c[r_] = A[torch.as_tensor(others[pick], device=dev)].mean(0)
            ctxs[k] = c

        out = {"training_ctx_norm": target, "ctx_scale": scale, "norms": {}, "repairs": {}}
        for k in a.prefix:
            n = ctxs[k].norm(dim=1)
            out["norms"][f"k={k}"] = float(n[n > 0].mean()) if k else 0.0

        # Three ways of handling a partial cart:
        #   raw            use the prefix mean as-is
        #   zero if k<3    ignore the cart until it is big enough
        #   norm-matched   rescale the prefix mean to the norm training produced
        for rep in ["raw", "zero if k<3", "norm-matched"]:
            row = {}
            for k in a.prefix:
                c = ctxs[k].clone()
                if rep == "zero if k<3" and 0 < k < 3:
                    c = torch.zeros_like(c)
                if rep == "norm-matched" and k > 0:
                    nn = c.norm(dim=1, keepdim=True).clamp_min(1e-6)
                    c = c * (target / nn)
                ll, top1 = score(m, d, dev, rows, cand, avail, c, dlogp, st, wt, stt, ut)
                row[f"k={k}"] = {"loglik": ll, "top1": top1}
            out["repairs"][rep] = row
        ll, top1 = score(m, d, dev, rows, cand, avail, full, dlogp, st, wt, stt, ut)
        out["full basket"] = {"loglik": ll, "top1": top1}
        res["models"][label] = out
        log(f"  {label:16s} ||ctx|| train {target:.3f}, scale {scale:.2f}; "
            + "  ".join(f"k={k}:{out['norms'][f'k={k}']/max(target,1e-9):.2f}x"
                        for k in a.prefix if k))
        for rep, row in out["repairs"].items():
            log(f"    {rep:14s}" + "".join(f"{row[f'k={k}']['loglik']:9.4f}"
                                           for k in a.prefix)
                + f"   full {out['full basket']['loglik']:.4f}")

    # ---- a baseline that needs no basket at all, on the same candidate sets
    tr = d.splits["train"]
    Hm = np.zeros((d.N, d.J), dtype=np.float32)
    np.add.at(Hm, (tr["user"], tr["item"]), 1.0)
    Hm = np.log1p(Hm)
    # tuned on VALIDATION, as every baseline in this repository is
    vsp = d.splits["validation"]
    vr = np.random.default_rng(a.seed + 1)
    vb = vr.choice(vsp["n_baskets"], size=min(1500, vsp["n_baskets"]), replace=False)
    vrows = np.concatenate([np.arange(vsp["starts"][i], vsp["ends"][i]) for i in vb])
    vu, vi = vsp["user"][vrows], vsp["item"][vrows]
    vc = np.concatenate([vi[:, None],
                         vr.choice(d.J, size=(len(vrows), a.n_neg), p=d.neg_p)], axis=1)
    v0 = torch.as_tensor(Hm[vu[:, None], vc], device=dev)
    best = max(((float(torch.log_softmax(v0 * w, dim=1)[:, 0].mean()), w)
                for w in a.grid))
    s = torch.as_tensor(Hm[user[:, None], cand_np], device=dev) * best[1]
    lp = torch.log_softmax(s, dim=1)[:, 0]
    res["household_repeat_purchase"] = {"loglik": float(lp.mean()), "weight": best[1],
                                        "top1": float((s.argmax(1) == 0).float().mean())}
    log(f"  {'household repeat':18s} loglik {res['household_repeat_purchase']['loglik']:.4f} "
        f"(needs no basket)")

    with open(os.path.join(OUT, "context_ablation.json"), "w") as f:
        json.dump(res, f, indent=2)

    # ------------------------------------------------------------------- figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.5, 5))
    xs = list(a.prefix) + [max(a.prefix) + 2]
    for lb, col in zip(res["models"], [PAL["blue"], PAL["green"], PAL["red"], PAL["grey"]]):
        o = res["models"][lb]
        ys = [o["repairs"]["raw"][f"k={k}"]["loglik"] for k in a.prefix] + \
             [o["full basket"]["loglik"]]
        ax.plot(xs, ys, "o-", color=col, lw=2, label=lb)
    ax.axhline(res["household_repeat_purchase"]["loglik"], color="k", ls="--", lw=1.2,
               label="household repeat-purchase (no basket)")
    ax.set_xticks(xs)
    ax.set_xticklabels([str(k) for k in a.prefix] + ["all"])
    ax.set_xlabel("items already known to be in the basket")
    ax.set_ylabel("held-out item log-likelihood")
    ax.set_title("What the model is worth when the basket is not yet known", fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "context_ablation.png"), dpi=150, bbox_inches="tight")
    log("")
    log("wrote out/context_ablation.json and figures/context_ablation.png")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--labels", nargs="+",
                   default=["nested", "nested_noctx", "nested_both", "nested_hn75"])
    p.add_argument("--prefix", type=int, nargs="+", default=[0, 1, 2, 4, 8])
    p.add_argument("--n-baskets", type=int, default=4000)
    p.add_argument("--n-neg", type=int, default=20)
    p.add_argument("--grid", type=float, nargs="+",
                   default=[0.25, 0.5, 1, 2, 3, 5, 8])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    main(p.parse_args())
