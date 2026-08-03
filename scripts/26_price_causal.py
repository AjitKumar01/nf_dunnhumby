"""
Stage 26 -- Is the basket model's price coefficient causal enough to answer what-if?

25_basket_placebo.py tests the *data*: it asks whether the price variation in the
188-category catalogue is clean, using a reduced form.  This script tests the *model*:
whether the fitted price parameter is measuring price, and whether the model predicts
better in exactly the weeks where price moved.

Three checks.

1. Structural placebo.  The same model is refitted on a scrambled price panel --
   each item's own days reordered ("permute"), or each item given another item's
   series ("swap").  The fitted price coefficient must collapse.  If it survives, it
   was fitting something other than price and no counterfactual from it is safe.
   This is stronger than the reduced-form placebo, because everything else in the
   model is free to re-adjust and can in principle compensate.

2. Counterfactual-relevant fit.  Held-out log-likelihood restricted to item-weeks in
   which the item's price actually moved.  A model can look fine on average and be
   useless exactly where the question lives; this is the metric the paper selects on.

3. Elasticity, clean subset against the rest.  Own-price elasticity computed from the
   fitted parameters, split by the per-category verdict from 25.  If the categories
   that fail a placebo carry systematically larger elasticities, the average is being
   inflated by endogeneity.

Writes out/price_causal.json and figures/price_causal.png.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import torch

import importlib
bm = importlib.import_module("23_basket_model")

HERE = os.path.dirname(os.path.abspath(__file__))
IN = os.path.join(HERE, "..", "basket_input")
OUT = os.path.join(HERE, "..", "out")
FIG = os.path.join(HERE, "..", "figures")

PALETTE = {"blue": "#2d6cdf", "grey": "#9aa5b1", "red": "#d1495b",
           "green": "#2a9d8f", "amber": "#e9c46a"}


def log(m):
    print(f"[26] {m}", flush=True)


def load(label, d, n_weeks, dev):
    cfg = json.load(open(os.path.join(OUT, f"{label}_basket_history.json")))["config"]
    m = bm.BasketModel(d, K=cfg["K"], Kp=cfg["Kp"],
                       use_context=not cfg["no_context"],
                       use_state=not cfg["no_state"],
                       use_price=not cfg["no_price"],
                       use_taste=not cfg["no_taste"],
                       seed=cfg["seed"], tie_context=cfg.get("tie_context", False),
                       Kt=0 if cfg.get("no_season") else cfg.get("Kt", 0),
                       n_weeks=n_weeks).to(dev)
    m.load_state_dict(torch.load(os.path.join(OUT, f"{label}_basket.pt"),
                                 map_location=dev))
    m.eval()
    return m, cfg


def price_coef(m):
    """gamma_i . beta_j over all households and items -> the price coefficient."""
    if m.gamma is None:
        return None
    with torch.no_grad():
        return (m.gamma @ m.beta.T).cpu().numpy()          # [N, J]


@torch.no_grad()
def loglik_on(m, d, split, mask_rows, n_neg, seed, dev, max_rows=200000):
    """Mean log-likelihood over a chosen subset of held-out rows.

    Scored basket by basket so the context term is exact, then filtered to the rows
    the mask selects -- filtering first would change every basket's context and score
    a different model from the one that was fitted.
    """
    sp = d.splits[split]
    rng = np.random.default_rng(seed)
    keep_b = np.flatnonzero(np.add.reduceat(mask_rows.astype(np.int64),
                                            sp["starts"]) > 0)
    if len(keep_b) == 0:
        return np.nan, 0
    tot, cnt = 0.0, 0
    alpha_det = m.alpha.detach()
    for a in range(0, len(keep_b), 256):
        b = keep_b[a:a + 256]
        users, cand, ctx, dlogp, st, wk = bm.make_batch(
            d, split, b, n_neg, rng, m.K, dev, alpha_det)
        s = m.score(users, cand, ctx, dlogp, st, wk)
        lp = torch.log_softmax(s, dim=1)[:, 0].cpu().numpy()
        rows = np.concatenate([np.arange(sp["starts"][i], sp["ends"][i]) for i in b])
        sel = mask_rows[rows]
        tot += float(lp[sel].sum()); cnt += int(sel.sum())
        if cnt > max_rows:
            break
    return tot / max(cnt, 1), cnt


def main(a):
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(FIG, exist_ok=True)
    dev = torch.device(a.device)
    d = bm.BasketData(IN, device=dev)
    n_weeks = int(max(v["week"].max() for v in d.splits.values())) + 1
    items = pd.read_parquet(os.path.join(IN, "items.parquet"))

    res = {}

    # ------------------------------------------- 1. structural placebo
    log("1. structural placebo: does the fitted price coefficient survive scrambling?")
    coefs = {}
    for label in a.labels:
        if not os.path.exists(os.path.join(OUT, f"{label}_basket.pt")):
            log(f"   {label}: no checkpoint, skipping")
            continue
        m, cfg = load(label, d, n_weeks, dev)
        C = price_coef(m)
        if C is None:
            continue
        coefs[label] = C
        hist = json.load(open(os.path.join(OUT, f"{label}_basket_history.json")))
        res[label] = {
            "placebo_price": cfg.get("placebo_price", "none"),
            "seasonality_rank": 0 if cfg.get("no_season") else cfg.get("Kt", 0),
            "median_price_coef": float(np.median(C)),
            "mean_price_coef": float(C.mean()),
            "sd_price_coef_across_items": float(C.mean(0).std()),
            "share_positive": float((C > 0).mean()),
            "test_loglik": hist["test_loglik"],
        }
        r = res[label]
        log(f"   {label:14s} placebo={r['placebo_price']:8s} "
            f"median gamma.beta {r['median_price_coef']:+.4f}  "
            f"sd across items {r['sd_price_coef_across_items']:.4f}  "
            f"test loglik {r['test_loglik']:.4f}")

    real = a.labels[0]
    if real in res:
        base = res[real]["median_price_coef"]
        for label in a.labels[1:]:
            if label in res and res[label]["placebo_price"] != "none":
                ratio = res[label]["median_price_coef"] / base if base else np.nan
                res[label]["share_of_real_coefficient_retained"] = float(ratio)
                log(f"   -> {label} retains {ratio:.1%} of the real coefficient")

    # ---------------------------------- 2. fit where the question lives
    log("")
    log("2. held-out fit restricted to item-weeks where the price actually moved")
    logp = np.load(os.path.join(IN, "log_price.npy"))
    bk = pd.read_parquet(os.path.join(IN, "baskets.parquet"))
    d2w = bk[["DAY", "WEEK_NO"]].drop_duplicates().set_index("DAY").WEEK_NO
    wk_of_day = d2w.reindex(np.arange(logp.shape[1])).ffill().bfill().to_numpy().astype(int)
    # weekly mean log price per item, then the week-on-week change
    weeks = np.unique(wk_of_day)
    wp = np.stack([logp[:, wk_of_day == w].mean(1) for w in weeks], axis=1)  # [J, W]
    dwp = np.abs(np.diff(wp, axis=1, prepend=wp[:, :1]))
    moved = dwp > a.move_tol                                               # [J, W]
    w_index = {w: i for i, w in enumerate(weeks)}
    sp = d.splits["test"]
    row_week_ix = np.array([w_index.get(w, 0) for w in sp["week"]])
    mask_moved = moved[sp["item"], row_week_ix]
    log(f"   {int(mask_moved.sum()):,} of {len(mask_moved):,} held-out rows are in an "
        f"item-week where the price moved by >{a.move_tol}")
    for label in a.labels:
        if label not in res:
            continue
        m, _ = load(label, d, n_weeks, dev)
        ll_m, n_m = loglik_on(m, d, "test", mask_moved, a.n_neg, 999, dev)
        ll_s, n_s = loglik_on(m, d, "test", ~mask_moved, a.n_neg, 999, dev)
        res[label]["test_loglik_price_moved"] = ll_m
        res[label]["test_loglik_price_static"] = ll_s
        res[label]["n_moved"], res[label]["n_static"] = int(n_m), int(n_s)
        log(f"   {label:14s} price-move weeks {ll_m:.4f} ({n_m:,})   "
            f"static weeks {ll_s:.4f} ({n_s:,})")

    # ------------------------------- 3. elasticity on clean vs failing categories
    log("")
    log("3. elasticity by placebo verdict")
    vpath = os.path.join(OUT, "basket_placebo_clean_categories.csv")
    if os.path.exists(vpath) and real in coefs:
        V = pd.read_csv(vpath)
        C = coefs[real]
        item_coef = C.mean(0)                                    # [J]
        it = items.sort_values("item_id")
        df = pd.DataFrame({"item_id": it.item_id.to_numpy(),
                           "category": it.COMMODITY_DESC.to_numpy(),
                           "coef": item_coef})
        df = df.merge(V[["category", "clean", "usable", "fails_strict_placebo",
                         "real_significant_negative"]], on="category", how="left")
        scored = df.dropna(subset=["clean"])
        grp = {
            "clean (passes every placebo)": scored[scored.clean == True],
            "usable (no strict failure)": scored[scored.usable == True],
            "fails a strict placebo": scored[scored.fails_strict_placebo == True],
            "not scored / too small": df[df.clean.isna()],
        }
        res["elasticity_by_verdict"] = {}
        for k, g in grp.items():
            if not len(g):
                continue
            res["elasticity_by_verdict"][k] = {
                "items": int(len(g)),
                "median_price_coef": float(g.coef.median()),
                "mean_price_coef": float(g.coef.mean()),
            }
            log(f"   {k:32s} {len(g):5,} items   median gamma.beta "
                f"{g.coef.median():+.4f}")
        df.to_csv(os.path.join(OUT, "price_coef_by_item.csv"), index=False)

    with open(os.path.join(OUT, "price_causal.json"), "w") as f:
        json.dump(res, f, indent=2, default=float)

    # ------------------------------------------------------------------ figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    have = [l for l in a.labels if l in res]
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.0))

    ax = axes[0]
    names = [l for l in have]
    vals = [res[l]["median_price_coef"] for l in names]
    cols = [PALETTE["blue"] if res[l]["placebo_price"] == "none" else PALETTE["red"]
            for l in names]
    ax.barh(range(len(names)), vals, color=cols)
    ax.axvline(0, color="k", lw=1)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([f"{l}\n({res[l]['placebo_price']})" for l in names], fontsize=8)
    ax.set_xlabel("median fitted price coefficient  $\\gamma_i\\cdot\\beta_j$")
    ax.set_title("Structural placebo\nred = price panel scrambled before fitting",
                 fontsize=10)
    ax.grid(axis="x", alpha=.3)
    for i, v in enumerate(vals):
        ax.text(v, i, f"  {v:+.3f}", va="center", fontsize=8)

    ax = axes[1]
    lm = [res[l].get("test_loglik_price_moved", np.nan) for l in names]
    ls = [res[l].get("test_loglik_price_static", np.nan) for l in names]
    x = np.arange(len(names))
    ax.barh(x - .2, lm, height=.38, color=PALETTE["blue"], label="price moved")
    ax.barh(x + .2, ls, height=.38, color=PALETTE["grey"], label="price static")
    ax.set_yticks(x); ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("held-out log-likelihood")
    ax.set_title("Fit where the counterfactual lives", fontsize=10)
    ax.legend(fontsize=8); ax.grid(axis="x", alpha=.3)

    ax = axes[2]
    ev = res.get("elasticity_by_verdict", {})
    if ev:
        k = list(ev.keys())
        v = [ev[i]["median_price_coef"] for i in k]
        ax.barh(range(len(k)), v, color=PALETTE["green"])
        ax.set_yticks(range(len(k)))
        ax.set_yticklabels([f"{i}\n({ev[i]['items']:,} items)" for i in k], fontsize=7)
        ax.set_xlabel("median price coefficient")
        ax.set_title("Is the elasticity inflated by\nthe categories that fail a placebo?",
                     fontsize=10)
        ax.grid(axis="x", alpha=.3)
        for i, val in enumerate(v):
            ax.text(val, i, f"  {val:+.3f}", va="center", fontsize=8)

    fig.suptitle("Can the basket model answer what-if questions about price?", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "price_causal.png"), dpi=150, bbox_inches="tight")
    log("")
    log("wrote out/price_causal.json and figures/price_causal.png")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--labels", nargs="+",
                   default=["season", "season_pl_p", "season_pl_s", "season_nos"])
    p.add_argument("--n-neg", type=int, default=20)
    p.add_argument("--move-tol", type=float, default=0.02,
                   help="log-price change that counts as the price having moved")
    p.add_argument("--device", default="cpu")
    main(p.parse_args())
