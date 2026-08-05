"""
Stage 32 -- How reliably can one household's price sensitivity be measured?

DATA_EXPLORATION.md 7.2 reported a split-half correlation of +0.236 and read it as
"roughly a quarter of the apparent spread is real, the rest is noise".  That reading
was asserted, not tested.  Testing it changes the conclusion in two ways, so the tests
now live here rather than in a sentence.

The question matters specifically for personalised couponing.  Targeting has to *rank*
households by price sensitivity, so what counts is not whether sensitivity differs on
average across the population, but whether an individual household's estimate is
stable enough to act on.  Split-half reliability answers exactly that: measure each
household twice on disjoint halves of its own history and see whether the two agree.

Four things are measured:

  1. is the agreement real?           against a null that shuffles household labels
  2. what about the FULL history?      split-half uses half the data per side, which
                                       understates it; Spearman-Brown corrects for it
  3. noise-limited or truly alike?     reliability against how much data each
                                       household provides.  If households were simply
                                       similar, more data would not help.  If the
                                       limit is estimation noise, it would.
  4. does it support targeting?        rank on one half, then check whether those
                                       groups actually differ in the other half

Writes out/reliability.json and figures/reliability.png.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
IN = os.path.join(HERE, "..", "basket_input")
OUT = os.path.join(HERE, "..", "out")
FIG = os.path.join(HERE, "..", "figures")
PAL = {"blue": "#2d6cdf", "grey": "#9aa5b1", "red": "#d1495b", "green": "#2a9d8f"}


def log(m):
    print(f"[32] {m}", flush=True)


def slopes(df, min_obs):
    """Per-household price slope, demeaned within (household, item).

    Identical to the estimator in 30_household_eda.py, so the numbers here are
    directly comparable with DATA_EXPLORATION.md 7.2.
    """
    g = df[df.groupby(["user_id", "item_id"]).item_id.transform("size") >= 3].copy()
    k = ["user_id", "item_id"]
    g["yd"] = g.lu - g.groupby(k).lu.transform("mean")
    g["xd"] = g.lp - g.groupby(k).lp.transform("mean")
    a = g.assign(xy=g.xd * g.yd, xx=g.xd ** 2).groupby("user_id").agg(
        xy=("xy", "sum"), xx=("xx", "sum"), n=("xx", "size"))
    a = a[(a.n >= min_obs) & (a.xx > 1e-6)]
    return (a.xy / a.xx).rename("slope"), a.n.rename("rows")


def main(a):
    os.makedirs(FIG, exist_ok=True)
    bk = pd.read_parquet(os.path.join(IN, "baskets.parquet"))
    logp = np.load(os.path.join(IN, "log_price.npy"))
    bk = bk.sort_values(["user_id", "DAY"])
    bk["lp"] = logp[bk.item_id.to_numpy(), bk.DAY.to_numpy()]
    bk["lu"] = np.log(bk.units.to_numpy(float))
    med = bk.groupby("user_id").DAY.transform("median")
    first, second = bk[bk.DAY <= med], bk[bk.DAY > med]

    r = {}
    s1, n1 = slopes(first, a.min_obs)
    s2, n2 = slopes(second, a.min_obs)
    b = pd.concat([s1.rename("h1"), s2.rename("h2"),
                   n1.rename("n1"), n2.rename("n2")], axis=1).dropna()
    rho = float(b.h1.corr(b.h2))
    rho_s = float(b.h1.corr(b.h2, method="spearman"))

    # ---------------------------------------------------------- 1. is it real?
    rng = np.random.default_rng(a.seed)
    null = np.array([b.h1.corr(pd.Series(rng.permutation(b.h2.values), index=b.index))
                     for _ in range(a.n_perm)])
    r["is_it_real"] = {
        "households": int(len(b)), "pearson": rho, "spearman": rho_s,
        "null_mean": float(null.mean()), "null_sd": float(null.std()),
        "sd_above_null": float((rho - null.mean()) / null.std()),
        "p_value_upper_bound": float((null >= rho).mean() + 1 / a.n_perm),
    }
    q = r["is_it_real"]
    log(f"1. real?  r = {rho:+.4f} (Spearman {rho_s:+.4f}) on {len(b):,} households")
    log(f"   null from shuffling household labels: {null.mean():+.4f} +/- {null.std():.4f}")
    log(f"   -> {q['sd_above_null']:.1f} sd above the null, p < {q['p_value_upper_bound']:.3f}")

    # -------------------------------------------------- 2. what the full history gives
    sb = 2 * rho / (1 + rho)
    r["full_history"] = {"split_half": rho, "spearman_brown": float(sb)}
    log("")
    log("2. split-half halves the data on each side, so it understates the reliability")
    log(f"   of an estimate built on everything.  Spearman-Brown 2r/(1+r) = {sb:+.4f}")

    # ------------------------------------------ 3. noise-limited or truly alike?
    rows = []
    for m in a.thresholds:
        x1, _ = slopes(first, m)
        x2, _ = slopes(second, m)
        bb = pd.concat([x1.rename("h1"), x2.rename("h2")], axis=1).dropna()
        if len(bb) > 30:
            rr = float(bb.h1.corr(bb.h2))
            rows.append({"min_rows_per_half": m, "households": int(len(bb)),
                         "r": rr, "spearman_brown": float(2 * rr / (1 + rr))})
    r["by_data_volume"] = rows
    log("")
    log("3. reliability against how much data each household provides:")
    for x in rows:
        log(f"   >= {x['min_rows_per_half']:4d} rows/half: r = {x['r']:+.4f}  "
            f"(implied full-history {x['spearman_brown']:+.4f})  "
            f"on {x['households']:,} households")
    log("   -> rising steeply with data means the binding constraint is MEASUREMENT")
    log("      NOISE, not households being genuinely alike.")

    # ------------------------------------------------- 4. does it support targeting?
    b2 = b.copy()
    b2["tercile"] = pd.qcut(b2.h1, 3, labels=["most sensitive", "middle", "least"])
    g = b2.groupby("tercile", observed=True).h2.agg(["mean", "median", "size"])
    spread = float(g.loc["most sensitive", "mean"] - g.loc["least", "mean"])
    sd = float(b2.h2.std())
    r["targeting"] = {
        "tercile_means_in_held_out_half": {str(k): float(v) for k, v in g["mean"].items()},
        "spread_most_vs_least": spread, "pooled_sd": sd,
        "spread_in_sd_units": spread / sd,
    }
    log("")
    log("4. targeting test -- rank households on the first half, measure on the second:")
    for k, v in g["mean"].items():
        log(f"   {str(k):16s} held-out slope {v:+.4f}   ({int(g.loc[k, 'size']):,} households)")
    log(f"   spread most vs least: {spread:+.4f} = {abs(spread) / sd:.2f} sd")

    with open(os.path.join(OUT, "reliability.json"), "w") as f:
        json.dump(r, f, indent=2)

    # ------------------------------------------------------------------- figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))

    ax = axes[0]
    ax.hist(null, bins=30, color=PAL["grey"], edgecolor="white",
            label=f"household labels shuffled\n(mean {null.mean():+.3f})")
    ax.axvline(rho, color=PAL["red"], lw=2.5,
               label=f"actual {rho:+.3f}\n{q['sd_above_null']:.0f} sd above the null")
    ax.set_xlabel("correlation between a household's two half-histories")
    ax.set_ylabel("shuffles")
    ax.set_title("Is the agreement real?", fontsize=10)
    ax.legend(fontsize=7)
    ax.grid(alpha=.3)

    ax = axes[1]
    xs = [x["min_rows_per_half"] for x in rows]
    ax.plot(xs, [x["r"] for x in rows], "o-", color=PAL["blue"], lw=2,
            label="split-half r")
    ax.plot(xs, [x["spearman_brown"] for x in rows], "s--", color=PAL["green"], lw=2,
            label="implied full-history")
    ax.axhline(0, color="k", lw=.8)
    ax.set_xscale("log")
    ax.set_xlabel("minimum purchase rows per half-history")
    ax.set_ylabel("reliability")
    ax.set_title("Rising with data means noise-limited,\nnot households being alike",
                 fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=.3)
    for x in rows:
        ax.annotate(f"n={x['households']:,}", (x["min_rows_per_half"], x["r"]),
                    fontsize=6, textcoords="offset points", xytext=(0, -14), ha="center")

    ax = axes[2]
    ks = list(g.index)
    ax.bar(range(len(ks)), [g.loc[k, "mean"] for k in ks],
           color=[PAL["blue"], PAL["grey"], PAL["red"]])
    ax.axhline(0, color="k", lw=.8)
    ax.set_xticks(range(len(ks)))
    ax.set_xticklabels([str(k) for k in ks], fontsize=8)
    ax.set_ylabel("price slope in the held-out half")
    ax.set_title(f"Ranked on one half, measured on the other\n"
                 f"spread {abs(spread):.2f} = {abs(spread) / sd:.2f} sd", fontsize=10)
    ax.grid(axis="y", alpha=.3)
    for i, k in enumerate(ks):
        ax.text(i, g.loc[k, "mean"], f"{g.loc[k, 'mean']:+.3f}", ha="center",
                va="bottom" if g.loc[k, "mean"] >= 0 else "top", fontsize=8)

    fig.suptitle("Can one household's price sensitivity be measured well enough to target on?",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "reliability.png"), dpi=150, bbox_inches="tight")
    log("")
    log("wrote out/reliability.json and figures/reliability.png")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--min-obs", type=int, default=40,
                   help="purchase rows a household needs in EACH half to be included")
    p.add_argument("--thresholds", type=int, nargs="+", default=[20, 40, 80, 160, 320])
    p.add_argument("--n-perm", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    main(p.parse_args())
