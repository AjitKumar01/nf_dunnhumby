"""
Stage 43 -- Confidence intervals for the ablation gaps and the elasticity.

Every number reported so far is a point estimate, with a two-seed spread standing in for
uncertainty.  That is not enough to call any gap real.

A full bootstrap would refit the model on each resample, which at ~25 min per fit is out
of reach.  But the dominant uncertainty in a HELD-OUT score is the test sample, and that
can be had almost free: score every basket once per model, then resample HOUSEHOLDS with
replacement and recompute the mean.  Households are the resampling unit because trips
repeat within a household and are not independent.

  What this covers      sampling variability of the held-out evaluation
  What it does NOT      variability from refitting -- optimisation noise, initialisation,
                        and the training sample.  The two-seed spread is the only handle
                        on that and it is reported alongside.

Gaps are computed PAIRED: the same resampled households score both models, so the
interval is on the difference, not the difference of two intervals.

Writes out/bootstrap.json.
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
    print(f"[43] {m}", flush=True)


@torch.no_grad()
def per_basket_scores(model, d, bidx, device, chunk=256):
    """Mean item log-likelihood per basket, and the own-price elasticity per purchase."""
    sp = d.splits["test"]
    ll, el, own = [], [], []
    for s in range(0, len(bidx), chunk):
        b = bidx[s:s + chunk]
        bt = nb.make_batch(d, model, "test", b, np.random.default_rng(0), device)
        u = model.item_utility(bt["user"], bt["cand"], bt["ctx"], bt["dlogp"],
                               bt["state"], bt["week"], bt["store"])
        u = u.masked_fill(~bt["avail"], -1e9)
        ar = torch.arange(u.shape[0], device=device)
        lp = torch.log_softmax(u, 1)[ar, bt["target"]]
        pi = torch.softmax(u, 1)[ar, bt["target"]]
        j = bt["cand"][ar, bt["target"]]
        gb = (model.gamma[bt["user"]] * model.beta[j]).sum(-1)
        e = -(gb * (1 - pi))
        ow = bt["owner"].cpu().numpy()
        for k in range(len(b)):
            m_ = ow == k
            if m_.sum():
                ll.append(float(lp[torch.as_tensor(m_, device=device)].mean()))
                el.append(float(e[torch.as_tensor(m_, device=device)].mean()))
                own.append(int(sp["user"][sp["starts"][b[k]]]))
    return np.array(ll), np.array(el), np.array(own)


def main(a):
    dev = torch.device("cpu")
    d = nb.NestedData(IN, device=dev)
    sp = d.splits["test"]
    rng = np.random.default_rng(a.seed)
    bidx = rng.choice(sp["n_baskets"], size=min(a.n_baskets, sp["n_baskets"]),
                      replace=False)

    log(f"scoring {len(bidx):,} held-out baskets once per model")
    S, E, HH = {}, {}, None
    for lab in a.labels:
        if not os.path.exists(os.path.join(OUT, f"{lab}_nested.pt")):
            log(f"  {lab}: no checkpoint, skipping")
            continue
        m, _ = cf.load(lab, d, dev)
        ll, el, own = per_basket_scores(m, d, bidx, dev)
        S[lab], E[lab] = ll, el
        HH = own
        log(f"  {lab:20s} item {ll.mean():+.4f}  elasticity {np.median(el):+.4f}")

    hh = np.unique(HH)
    idx_by_hh = {h: np.flatnonzero(HH == h) for h in hh}
    log(f"\n{len(hh):,} households across those baskets; "
        f"{a.reps:,} household block-bootstrap resamples")

    def resample(r):
        pick = r.choice(hh, size=len(hh), replace=True)
        return np.concatenate([idx_by_hh[h] for h in pick])

    r = np.random.default_rng(a.seed + 1)
    draws = [resample(r) for _ in range(a.reps)]

    base = a.labels[0]
    res = {"n_baskets": int(len(bidx)), "n_households": int(len(hh)),
           "reps": a.reps, "base": base, "models": {}}
    log("")
    log(f"  {'model':20s} {'item':>9s} {'95% CI':>18s} {'gap vs full':>12s} "
        f"{'95% CI on gap':>20s}")
    for lab in S:
        bs = np.array([S[lab][ix].mean() for ix in draws])
        lo, hi = np.percentile(bs, [2.5, 97.5])
        row = {"item": float(S[lab].mean()), "ci": [float(lo), float(hi)]}
        if lab != base:
            gd = np.array([S[base][ix].mean() - S[lab][ix].mean() for ix in draws])
            glo, ghi = np.percentile(gd, [2.5, 97.5])
            row["gap"] = float(S[base].mean() - S[lab].mean())
            row["gap_ci"] = [float(glo), float(ghi)]
            sig = "" if glo * ghi > 0 else "   <- CI spans 0"
            log(f"  {lab:20s} {S[lab].mean():+9.4f} [{lo:+7.4f},{hi:+7.4f}]"
                f" {row['gap']:+12.4f} [{glo:+8.4f},{ghi:+8.4f}]{sig}")
        else:
            log(f"  {lab:20s} {S[lab].mean():+9.4f} [{lo:+7.4f},{hi:+7.4f}]"
                f" {'--':>12s} {'--':>20s}")
        res["models"][lab] = row

    log("")
    log("  own-price elasticity (allocation margin), median over purchases")
    for lab in E:
        bs = np.array([np.median(E[lab][ix]) for ix in draws])
        lo, hi = np.percentile(bs, [2.5, 97.5])
        res["models"][lab]["elasticity"] = float(np.median(E[lab]))
        res["models"][lab]["elasticity_ci"] = [float(lo), float(hi)]
        log(f"  {lab:20s} {np.median(E[lab]):+9.4f} [{lo:+7.4f},{hi:+7.4f}]")

    with open(os.path.join(OUT, "bootstrap.json"), "w") as f:
        json.dump(res, f, indent=2)
    log("")
    log("wrote out/bootstrap.json")
    log("NOTE: this is evaluation uncertainty only.  Refitting variability is not")
    log("      included; the two-seed spread is the only handle on that.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--labels", nargs="+",
                   default=["ps_nested", "ps_off", "ps_pl"])
    p.add_argument("--n-baskets", type=int, default=6000)
    p.add_argument("--reps", type=int, default=400)
    p.add_argument("--seed", type=int, default=0)
    main(p.parse_args())
