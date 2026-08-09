"""
Stage 25 -- Price-endogeneity placebo tests for the basket catalogue.

The basket model is supposed to answer what-if questions about price.  That claim
rests entirely on whether its price variation is exogenous, and nothing in this
repository has tested that on the 188-category catalogue -- the battery in
11_placebo_tests.py was built for the paper's 56-category Sunday/Monday sample and
tests a different identifying assumption.

The two designs identify price response from different variation, and the difference
matters:

  paper / nf     price changes at the Sunday->Monday week boundary, with a
                 category-week control.  Narrow, but the control absorbs anything
                 that moves demand at week frequency.
  basket model   *all* within-item price movement across 711 days, with no time
                 control whatsoever.  Wider, and therefore exposed to anything that
                 moves price and demand together -- holidays, seasons, and
                 promotions timed to demand.

So this script asks two questions, not one:

  1. does the price coefficient survive a placebo?
  2. how much of it is seasonality the model has no term for?

Question 2 is answered by running every test twice, once with and once without week
fixed effects.  If the coefficient is much larger without them, the basket model's
price parameter is partly a seasonality parameter and needs a time control before any
counterfactual is trustworthy.

Four placebos, from weakest to strongest null:

  forward / backward shift   the item's own price series moved by `shift_weeks`.
                             Weak, because a shifted series stays correlated with the
                             real one -- the paper's own rule has the same flaw
                             (PREPROCESSING.md 9).
  within-item permutation    the item's price series randomly reordered across weeks.
                             Destroys all time alignment, keeps the exact set of
                             prices the item really charged.
  cross-item swap            item j is given another item's price series from the
                             same category.  Destroys item alignment too.

Estimator: a within-transformed linear model of log purchase rate on log price, item
fixed effects always absorbed, week fixed effects optionally, standard errors
clustered by item.  This is a reduced form, not the structural model -- what it
measures is whether the *variation* is clean, which is a property of the data rather
than of the likelihood.  The structural counterpart is `23_basket_model.py
--placebo-price`, which refits the whole model on scrambled prices.

Writes out/basket_placebo.csv, out/basket_placebo_summary.json,
out/basket_placebo_clean_categories.csv and figures/basket_placebo.png.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
IN = os.path.join(HERE, "..", "..", "basket_input")
OUT = os.path.join(HERE, "..", "..", "out")
FIG = os.path.join(HERE, "..", "..", "figures")

PALETTE = {"blue": "#2d6cdf", "grey": "#9aa5b1", "red": "#d1495b",
           "green": "#2a9d8f", "amber": "#e9c46a"}


def log(m):
    print(f"[25] {m}", flush=True)


def absorb(X, groups_list):
    """Iteratively demean columns of X by each grouping (within transformation).

    Two-way fixed effects have no closed form here, so alternate the projections
    until they stop moving -- the standard Frisch-Waugh-Lovell approach.
    """
    X = X.astype(np.float64).copy()
    for _ in range(20):
        before = X.copy()
        for g in groups_list:
            n = np.bincount(g, minlength=g.max() + 1).astype(np.float64)
            for c in range(X.shape[1]):
                s = np.bincount(g, weights=X[:, c], minlength=len(n))
                X[:, c] -= (s / np.maximum(n, 1))[g]
        if np.max(np.abs(X - before)) < 1e-10:
            break
    return X


def fit_cluster(y, x, cluster):
    """Univariate OLS of y on x (both already demeaned) with clustered SEs."""
    xx = float(x @ x)
    if xx <= 1e-12:
        return np.nan, np.nan, np.nan, 0
    b = float(x @ y) / xx
    e = y - b * x
    # cluster-robust meat: sum over clusters of (sum_i x_i e_i)^2
    order = np.argsort(cluster)
    c_sorted, xe = cluster[order], (x * e)[order]
    bounds = np.flatnonzero(np.r_[True, np.diff(c_sorted) != 0])
    sums = np.add.reduceat(xe, bounds)
    meat = float((sums ** 2).sum())
    n_c = len(bounds)
    if n_c < 2:
        return b, np.nan, np.nan, n_c
    se = np.sqrt(meat) / xx
    # small-sample correction on the cluster count
    se *= np.sqrt(n_c / max(n_c - 1, 1))
    t = b / se if se > 0 else np.nan
    p = 2 * stats.t.sf(abs(t), df=max(n_c - 1, 1)) if np.isfinite(t) else np.nan
    return b, se, p, n_c


def build_panel(a):
    """Item x week purchases, exposure and log price."""
    bk = pd.read_parquet(os.path.join(IN, "baskets.parquet"))
    items = pd.read_parquet(os.path.join(IN, "items.parquet"))
    meta = json.load(open(os.path.join(IN, "meta.json")))
    logp = np.load(os.path.join(IN, "log_price.npy"))          # [J, D] raw log price

    # day -> week from the basket table itself, so the calendars cannot drift apart
    d2w = bk[["DAY", "WEEK_NO"]].drop_duplicates().set_index("DAY").WEEK_NO
    days = np.arange(logp.shape[1])
    wk_of_day = d2w.reindex(days).ffill().bfill().to_numpy().astype(int)

    # purchases per item-week
    y = (bk.groupby(["item_id", "WEEK_NO"]).size().rename("y").reset_index())
    # exposure: distinct shopping trips that week, the denominator of a purchase rate
    expo = bk.groupby("WEEK_NO").BASKET_ID.nunique().rename("trips")

    # price per item-week: mean log price over the week's days
    weeks = np.sort(bk.WEEK_NO.unique())
    wcols = {w: np.flatnonzero(wk_of_day == w) for w in weeks}
    rows = []
    for w in weeks:
        cols = wcols[w]
        if len(cols) == 0:
            continue
        rows.append(pd.DataFrame({"item_id": np.arange(logp.shape[0]),
                                  "WEEK_NO": w,
                                  "logp": logp[:, cols].mean(axis=1)}))
    P = pd.concat(rows, ignore_index=True)

    panel = P.merge(y, on=["item_id", "WEEK_NO"], how="left")
    panel["y"] = panel.y.fillna(0.0)
    panel = panel.merge(expo, on="WEEK_NO", how="left")
    panel = panel.merge(items[["item_id", "cat_id", "COMMODITY_DESC"]], on="item_id")
    # log purchase rate; the 0.5 keeps zero-purchase item-weeks in the sample rather
    # than silently conditioning on a positive outcome, which would be selection.
    panel["lry"] = np.log((panel.y + 0.5) / panel.trips)
    log(f"panel: {len(panel):,} item-weeks, {panel.item_id.nunique():,} items, "
        f"{panel.COMMODITY_DESC.nunique()} categories, {panel.WEEK_NO.nunique()} weeks; "
        f"{(panel.y > 0).mean():.1%} of item-weeks have a purchase")
    return panel


def make_placebo(panel, kind, rng, shift_weeks):
    """Return a copy of the log-price column under the given placebo."""
    p = panel.sort_values(["item_id", "WEEK_NO"]).reset_index(drop=True)
    g = p.groupby("item_id").logp
    if kind == "actual":
        out = p.logp.to_numpy()
    elif kind in ("forward", "backward"):
        k = shift_weeks if kind == "forward" else -shift_weeks
        out = g.transform(lambda s: np.roll(s.to_numpy(), k)).to_numpy()
    elif kind == "permute":
        # reorder each item's own price series across weeks
        out = g.transform(lambda s: rng.permutation(s.to_numpy())).to_numpy()
    elif kind == "swap":
        # give each item another item's series from the same category
        out = p.logp.to_numpy().copy()
        for cat, idx in p.groupby("cat_id").groups.items():
            sub = p.loc[idx]
            its = sub.item_id.unique()
            if len(its) < 2:
                continue
            perm = rng.permutation(len(its))
            # a derangement, so no item keeps its own series
            for shift in range(1, len(its)):
                if not np.any(perm == np.roll(np.arange(len(its)), shift)):
                    pass
            mapping = dict(zip(its, its[np.roll(np.arange(len(its)), 1)]))
            series = {i: sub.loc[sub.item_id == i].sort_values("WEEK_NO").logp.to_numpy()
                      for i in its}
            for i in its:
                m = (p.item_id == i)
                src = series[mapping[i]]
                tgt_n = int(m.sum())
                v = src if len(src) == tgt_n else np.resize(src, tgt_n)
                out[np.flatnonzero(m.to_numpy())] = v
    else:
        raise ValueError(kind)
    return p, out


def run(panel, kind, week_fe, rng, shift_weeks, min_items, min_obs):
    p, price = make_placebo(panel, kind, rng, shift_weeks)
    p = p.assign(price=price)
    rows = []
    for cat, g in p.groupby("COMMODITY_DESC"):
        if g.item_id.nunique() < min_items or len(g) < min_obs:
            continue
        item_code = pd.factorize(g.item_id)[0]
        week_code = pd.factorize(g.WEEK_NO)[0]
        groups = [item_code] + ([week_code] if week_fe else [])
        M = absorb(np.column_stack([g.lry.to_numpy(), g.price.to_numpy()]), groups)
        b, se, pv, nc = fit_cluster(M[:, 0], M[:, 1], item_code)
        rows.append({"category": cat, "placebo": kind, "week_fe": week_fe,
                     "coef": b, "se": se, "p": pv, "n_obs": len(g),
                     "n_items": int(g.item_id.nunique()), "n_clusters": nc})
    return pd.DataFrame(rows)


def main(a):
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(FIG, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    panel = build_panel(a)

    frames = []
    for week_fe in (False, True):
        for kind in ["actual", "forward", "backward", "permute", "swap"]:
            df = run(panel, kind, week_fe, np.random.default_rng(a.seed),
                     a.shift_weeks, a.min_items, a.min_obs)
            frames.append(df)
            sig = float((df.p < 0.01).mean()) if len(df) else np.nan
            log(f"  week_fe={str(week_fe):5s} {kind:9s}: {len(df):3d} categories, "
                f"median coef {df.coef.median():+.4f}, "
                f"mean {df.coef.mean():+.4f}, {sig:.1%} significant at 1%")
    R = pd.concat(frames, ignore_index=True)
    R.to_csv(os.path.join(OUT, "basket_placebo.csv"), index=False)

    # ------------------------------------------------------------------ summary
    summ = {}
    for week_fe in (False, True):
        key = "with_week_fe" if week_fe else "no_week_fe"
        s = {}
        for kind in ["actual", "forward", "backward", "permute", "swap"]:
            d = R[(R.placebo == kind) & (R.week_fe == week_fe)]
            if not len(d):
                continue
            ks = stats.kstest(d.p.dropna(), "uniform")
            s[kind] = {
                "categories": int(len(d)),
                "median_coef": float(d.coef.median()),
                "mean_coef": float(d.coef.mean()),
                "share_p_below_01": float((d.p < 0.01).mean()),
                "share_p_below_05": float((d.p < 0.05).mean()),
                "share_negative_and_sig01": float(((d.p < 0.01) & (d.coef < 0)).mean()),
                "ks_stat_vs_uniform": float(ks.statistic),
                "ks_p": float(ks.pvalue),
            }
        summ[key] = s

    # How much of the price coefficient is seasonality the model cannot see?
    a_no = R[(R.placebo == "actual") & (~R.week_fe)].set_index("category").coef
    a_fe = R[(R.placebo == "actual") & (R.week_fe)].set_index("category").coef
    both = pd.concat([a_no.rename("no_fe"), a_fe.rename("fe")], axis=1).dropna()
    summ["seasonality_share"] = {
        "categories": int(len(both)),
        "median_coef_without_week_fe": float(both.no_fe.median()),
        "median_coef_with_week_fe": float(both.fe.median()),
        "median_absolute_shrinkage": float((both.no_fe - both.fe).median()),
        "share_of_coefficient_removed_by_week_fe": float(
            1 - both.fe.median() / both.no_fe.median()) if both.no_fe.median() else np.nan,
    }
    ss = summ["seasonality_share"]
    log("")
    log(f"seasonality: median price coefficient {ss['median_coef_without_week_fe']:+.4f} "
        f"without week effects, {ss['median_coef_with_week_fe']:+.4f} with them "
        f"-> week effects remove {ss['share_of_coefficient_removed_by_week_fe']:.1%}")

    # ------------------------------------------------- per-category clean verdict
    # A category is clean if the real price effect is significantly negative and no
    # placebo produces a significant effect of the same sign.  Judged with week
    # fixed effects, which is the specification the model ought to have.
    fe = R[R.week_fe]
    piv = fe.pivot_table(index="category", columns="placebo", values=["coef", "p"])
    verdict = []
    for cat in piv.index:
        try:
            real_c, real_p = piv.loc[cat, ("coef", "actual")], piv.loc[cat, ("p", "actual")]
        except KeyError:
            continue
        fails = []
        for kind in ["forward", "backward", "permute", "swap"]:
            if ("coef", kind) not in piv.columns:
                continue
            c, pv = piv.loc[cat, ("coef", kind)], piv.loc[cat, ("p", kind)]
            if np.isfinite(pv) and pv < 0.01 and np.isfinite(c) and c < 0:
                fails.append(kind)
        verdict.append({
            "category": cat, "real_coef": real_c, "real_p": real_p,
            "real_significant_negative": bool(np.isfinite(real_p) and real_p < 0.01
                                              and real_c < 0),
            "n_placebo_failures": len(fails),
            "failed_placebos": ",".join(fails),
            "fails_strict_placebo": ("permute" in fails) or ("swap" in fails),
        })
    V = pd.DataFrame(verdict)
    V["clean"] = V.real_significant_negative & (V.n_placebo_failures == 0)
    V["usable"] = V.real_significant_negative & (~V.fails_strict_placebo)
    V.sort_values(["clean", "real_coef"]).to_csv(
        os.path.join(OUT, "basket_placebo_clean_categories.csv"), index=False)
    summ["verdict"] = {
        "categories": int(len(V)),
        "real_effect_significant_negative": int(V.real_significant_negative.sum()),
        "fail_at_least_one_placebo": int((V.n_placebo_failures > 0).sum()),
        "fail_a_strict_placebo": int(V.fails_strict_placebo.sum()),
        "clean": int(V.clean.sum()),
        "usable_passes_strict_only": int(V.usable.sum()),
    }
    v = summ["verdict"]
    log("")
    log(f"verdict on {v['categories']} categories: "
        f"{v['real_effect_significant_negative']} have a significantly negative real "
        f"price effect; {v['fail_at_least_one_placebo']} fail at least one placebo, "
        f"{v['fail_a_strict_placebo']} fail a strict one (permute or swap)")
    log(f"  -> {v['clean']} clean, {v['usable_passes_strict_only']} usable "
        f"(real effect present and no strict placebo failure)")

    with open(os.path.join(OUT, "basket_placebo_summary.json"), "w") as f:
        json.dump(summ, f, indent=2)

    # ------------------------------------------------------------------- figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.0))
    order = ["actual", "forward", "backward", "permute", "swap"]
    lbl = {"actual": "real prices", "forward": f"+{a.shift_weeks}w shift",
           "backward": f"-{a.shift_weeks}w shift", "permute": "weeks reordered",
           "swap": "another item's prices"}

    ax = axes[0]
    for week_fe, colour, nm in [(False, PALETTE["red"], "no week effects"),
                                (True, PALETTE["blue"], "with week effects")]:
        med = [R[(R.placebo == k) & (R.week_fe == week_fe)].coef.median() for k in order]
        ax.plot(range(len(order)), med, "o-", color=colour, lw=2, label=nm)
    ax.axhline(0, color="k", lw=1)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([lbl[k] for k in order], rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("median price coefficient across categories")
    ax.set_title("Does the price effect survive a fake price series?\n"
                 "a working placebo should sit at zero", fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=.3)

    ax = axes[1]
    w = 0.38
    for i, (week_fe, colour, nm) in enumerate(
            [(False, PALETTE["red"], "no week effects"),
             (True, PALETTE["blue"], "with week effects")]):
        sh = [float((R[(R.placebo == k) & (R.week_fe == week_fe)].p < 0.01).mean())
              for k in order]
        ax.bar(np.arange(len(order)) + (i - .5) * w, sh, width=w, color=colour, label=nm)
    ax.axhline(0.01, color="k", ls="--", lw=1, label="1% nominal")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([lbl[k] for k in order], rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("share of categories significant at 1%")
    ax.set_title("Over-rejection\nunder a true null this should be 1%", fontsize=10)
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=.3)

    ax = axes[2]
    if len(both):
        ax.scatter(both.no_fe, both.fe, s=16, alpha=.6, color=PALETTE["blue"])
        lim = [min(both.min().min(), -0.1), max(both.max().max(), 0.1)]
        ax.plot(lim, lim, "k--", lw=1, label="unchanged")
        ax.axhline(0, color="k", lw=.8); ax.axvline(0, color="k", lw=.8)
        ax.set_xlabel("price coefficient, no week effects")
        ax.set_ylabel("price coefficient, week effects")
        ax.set_title("How much is seasonality?\npoints above the line shrink toward zero",
                     fontsize=10)
        ax.legend(fontsize=8); ax.grid(alpha=.3)

    fig.suptitle("Price-endogeneity placebos on the 188-category basket catalogue",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "basket_placebo.png"), dpi=150, bbox_inches="tight")
    log("wrote out/basket_placebo.csv, basket_placebo_summary.json, "
        "basket_placebo_clean_categories.csv and figures/basket_placebo.png")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--shift-weeks", type=int, default=6)
    p.add_argument("--min-items", type=int, default=3)
    p.add_argument("--min-obs", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    main(p.parse_args())
