"""
Stage 34 -- Does the generated distribution match the real one?

NESTED_MODEL.md 8.6 compared three MEANS -- items, categories, units per basket -- and
8.6b added a mean pairwise alpha.alpha.  Neither is enough, and the second is close to
circular: alpha is the model's own embedding, and the model is trained to make
co-purchased items have large alpha.alpha, so scoring generated baskets with alpha
grades the exam with the answer key.

The object being learned is a basket -- a set of items with counts, conditional on
(household, day, week, store, prices).  A generator is only useful downstream if the
DISTRIBUTION matches, not three of its moments.  So every check here is

  (a) model-free      computed from item ids and counts, never from alpha, and
  (b) distributional  a whole distribution or a per-item vector, not a single mean.

Generated trips are matched to real held-out trips: same household, day, week, store,
so the conditioning information is identical and any gap is the model's.

  1. shape           full distributions of basket size, categories, units, breadth,
                     compared by total variation distance, not by their means
  2. item marginals  purchase rate of each of the 5,455 items, real vs generated
  3. co-occurrence   lift = P(j,k together) / (P(j) P(k)) for the commonest pairs.
                     This is the model-free version of the interaction term
  4. held-out labels share of within-basket pairs sharing a sub-commodity / department
                     -- labels the model never sees
  5. price response  rebuild the item x week panel FROM GENERATED DATA and run the
                     within-item elasticity of 29_demand_eda on it.  If a pricing
                     policy is going to be learned on generated baskets, the generated
                     baskets must carry the real price response.  This is the test that
                     matters most for the stated purpose and the one nothing ran before

Writes out/generator_eval.json and figures/generator_eval.png.
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
IN = os.path.join(HERE, "..", os.environ.get("NF_BASKET_INPUT", "basket_input"))
OUT = os.path.join(HERE, "..", "out")
FIG = os.path.join(HERE, "..", "figures")
PAL = {"blue": "#2d6cdf", "grey": "#9aa5b1", "red": "#d1495b", "green": "#2a9d8f"}


def log(m):
    print(f"[34] {m}", flush=True)


def tvd(a, b, bins):
    """Total variation distance between two count vectors on a shared support."""
    pa = np.histogram(a, bins=bins)[0].astype(float)
    pb = np.histogram(b, bins=bins)[0].astype(float)
    pa /= max(pa.sum(), 1)
    pb /= max(pb.sum(), 1)
    return 0.5 * np.abs(pa - pb).sum()


def within_item_elasticity(rows, logp, trips_per_week):
    """The 29_demand_eda estimator, on whatever (item, week, buyers) panel is given."""
    if not len(rows):
        return float("nan"), 0
    df = pd.DataFrame(rows, columns=["item_id", "week", "buyers"])
    df = df.groupby(["item_id", "week"], as_index=False).buyers.sum()
    df["trips"] = df.week.map(trips_per_week)
    df = df[df.trips > 0]
    df["lbuy"] = np.log((df.buyers + 0.5) / df.trips)
    df["logp"] = logp[df.item_id.to_numpy(), df.week.to_numpy()]
    df = df.dropna()
    n = df.groupby("item_id").item_id.transform("size")
    df = df[n >= 3]
    if len(df) < 100:
        return float("nan"), len(df)
    yd = df.lbuy - df.groupby("item_id").lbuy.transform("mean")
    xd = df.logp - df.groupby("item_id").logp.transform("mean")
    den = float((xd ** 2).sum())
    return (float((xd * yd).sum()) / den if den > 1e-9 else float("nan")), len(df)


def main(a):
    os.makedirs(FIG, exist_ok=True)
    dev = torch.device(a.device)
    d = nb.NestedData(IN, device=dev)
    m, _ = cf.load(a.label, d, dev)
    items = pd.read_parquet(os.path.join(IN, "items.parquet")).sort_values("item_id")
    sub = items.sub_id.to_numpy()
    dept = items.dept_id.to_numpy()
    sp = d.splits["test"]

    rng = np.random.default_rng(a.seed)
    n = min(a.n_trips, sp["n_baskets"])
    trips = rng.choice(sp["n_baskets"], size=n, replace=False)
    log(f"{n:,} held-out trips; generating with the SAME household, day, week and store")

    real = []
    for i in trips:
        r = np.arange(sp["starts"][i], sp["ends"][i])
        real.append((sp["item"][r].tolist(), sp["units"][r].astype(int).tolist(),
                     int(sp["raw_week"][r[0]])))
    weeks = np.array([x[2] for x in real])

    gens = {}
    for sw in a.sweeps:
        log(f"  generating with {sw} Gibbs sweeps ...")
        g = cf.generate_baskets(m, d, dev, n_trips=n, seed=a.seed, sweeps=sw,
                                use_ctx=sw > 0, with_units=True, trips=trips)
        gens[sw] = [(b[0], b[1], int(w)) for b, w in zip(g, weeks)]

    # ------------------------------------------------------------------ per-set stats
    def summarise(bk):
        sizes = np.array([len(b[0]) for b in bk])
        units = np.array([sum(b[1]) for b in bk])
        cats, brd, pairs_sub, pairs_dept, per_line = [], [], [], [], []
        for ids, us, _ in bk:
            c = d.item_cat[torch.as_tensor(ids, device=dev)].cpu().numpy() if len(ids) else np.array([])
            cats.append(len(set(c.tolist())))
            if len(c):
                brd += list(pd.Series(c).value_counts().to_numpy())
            per_line += list(us)
            if len(ids) > 1:
                s, dp_ = sub[ids], dept[ids]
                k = len(ids)
                pairs_sub.append(((s[:, None] == s[None, :]).sum() - k) / (k * (k - 1)))
                pairs_dept.append(((dp_[:, None] == dp_[None, :]).sum() - k) / (k * (k - 1)))
        return {"sizes": sizes, "units": units, "cats": np.array(cats),
                "breadth": np.array(brd), "per_line": np.array(per_line),
                "pair_sub": float(np.mean(pairs_sub)), "pair_dept": float(np.mean(pairs_dept))}

    S = {"real": summarise(real)}
    for sw in a.sweeps:
        S[f"gibbs{sw}"] = summarise(gens[sw])

    res = {"n_trips": int(n), "sweeps_tested": list(a.sweeps), "shape": {}}
    log("")
    log("1. shape -- whole distributions, compared by total variation distance")
    log(f"   {'':22s}" + "".join(f"{k:>14s}" for k in S))
    for name, key, bins in [("items per basket", "sizes", np.arange(0, 41)),
                            ("categories per basket", "cats", np.arange(0, 31)),
                            ("units per basket", "units", np.arange(0, 61)),
                            ("distinct items / cat", "breadth", np.arange(0, 9)),
                            ("units per line", "per_line", np.arange(0, 9))]:
        row = {}
        for k, v in S.items():
            row[k] = {"mean": float(v[key].mean()), "median": float(np.median(v[key])),
                      "p90": float(np.quantile(v[key], .9))}
            if k != "real":
                row[k]["tvd_vs_real"] = float(tvd(S["real"][key], v[key], bins))
        res["shape"][name] = row
        log(f"   {name:22s}" + "".join(
            f"{row[k]['mean']:8.2f}" + ("       " if k == "real"
                                        else f" ({row[k]['tvd_vs_real']:.2f})") for k in S))
    log("   (mean, and in brackets the total variation distance from the real "
        "distribution; 0 = identical, 1 = disjoint)")

    # ------------------------------------------------------------ 2. item marginals
    log("")
    log("2. per-item purchase rate across the whole catalogue")
    res["item_marginals"] = {}
    flat = lambda bk: np.asarray([j for b in bk for j in b[0]], dtype=np.int64)
    rc = np.bincount(flat(real), minlength=d.J)
    for k in [f"gibbs{s}" for s in a.sweeps]:
        sw = int(k[5:])
        gc = np.bincount(flat(gens[sw]), minlength=d.J)
        keep = (rc + gc) > 0
        pear = float(np.corrcoef(rc[keep], gc[keep])[0, 1])
        spear = float(pd.Series(rc[keep]).corr(pd.Series(gc[keep]), method="spearman"))
        res["item_marginals"][k] = {"pearson": pear, "spearman": spear,
                                    "items_compared": int(keep.sum()),
                                    "share_real_items_never_generated":
                                        float(((rc > 0) & (gc == 0)).sum() / max((rc > 0).sum(), 1))}
        v = res["item_marginals"][k]
        log(f"   {k:10s} pearson {pear:+.3f}  spearman {spear:+.3f}  "
            f"real items never generated {v['share_real_items_never_generated']:.1%}")

    # --------------------------------------------------------- 3. co-occurrence lift
    log("")
    log("3. co-occurrence lift -- the model-free version of the interaction term")

    def lifts(bk, top_items):
        idx = {j: q for q, j in enumerate(top_items)}
        K = len(top_items)
        co = np.zeros((K, K))
        solo = np.zeros(K)
        nb_ = 0
        for ids, _, _ in bk:
            u = sorted({j for j in ids if j in idx})
            if not u:
                continue
            nb_ += 1
            for q in u:
                solo[idx[q]] += 1
            for x in range(len(u)):
                for y in range(x + 1, len(u)):
                    co[idx[u[x]], idx[u[y]]] += 1
                    co[idx[u[y]], idx[u[x]]] += 1
        p = solo / max(nb_, 1)
        exp = np.outer(p, p) * max(nb_, 1)
        with np.errstate(divide="ignore", invalid="ignore"):
            L = np.where(exp > 0, co / np.maximum(exp, 1e-9), np.nan)
        return L, co

    top = np.argsort(-rc)[:a.top_items]
    Lr, Cr = lifts(real, top)
    iu = np.triu_indices(len(top), 1)
    res["cooccurrence"] = {"top_items": int(len(top))}
    for sw in a.sweeps:
        Lg, Cg = lifts(gens[sw], top)
        ok = np.isfinite(Lr[iu]) & np.isfinite(Lg[iu]) & ((Cr[iu] + Cg[iu]) >= a.min_pair)
        if ok.sum() < 30:
            continue
        sr = float(pd.Series(Lr[iu][ok]).corr(pd.Series(Lg[iu][ok]), method="spearman"))
        res["cooccurrence"][f"gibbs{sw}"] = {
            "pairs_compared": int(ok.sum()), "spearman_of_lift": sr,
            "mean_lift_real": float(np.mean(Lr[iu][ok])),
            "mean_lift_generated": float(np.mean(Lg[iu][ok]))}
        v = res["cooccurrence"][f"gibbs{sw}"]
        log(f"   gibbs{sw:<5d} {v['pairs_compared']:5,} pairs   "
            f"spearman of lift {sr:+.3f}   "
            f"mean lift real {v['mean_lift_real']:.2f} vs generated "
            f"{v['mean_lift_generated']:.2f}")

    # ------------------------------------------------------- 4. held-out label pairs
    log("")
    log("4. within-basket pairs sharing a held-out label (model never sees these)")
    res["held_out_labels"] = {k: {"same_sub": S[k]["pair_sub"], "same_dept": S[k]["pair_dept"]}
                              for k in S}
    for k in S:
        log(f"   {k:10s} same sub-commodity {S[k]['pair_sub']:.4f}   "
            f"same department {S[k]['pair_dept']:.4f}")

    # ------------------------------------------------------------- 5. price response
    log("")
    log("5. price response recovered FROM GENERATED DATA")
    logp_week = np.zeros((d.J, int(weeks.max()) + 1), dtype=np.float32)
    lp = np.load(os.path.join(IN, "log_price.npy"))
    bkall = pd.read_parquet(os.path.join(IN, "baskets.parquet"))
    dw = bkall.groupby("WEEK_NO").DAY.median().astype(int)
    for w, day in dw.items():
        if w < logp_week.shape[1]:
            logp_week[:, w] = lp[:, min(int(day), lp.shape[1] - 1)]
    tpw = pd.Series(weeks).value_counts().to_dict()

    def panel(bk):
        rows = []
        for ids, _, w in bk:
            for j in set(ids):
                rows.append((j, w, 1))
        return rows

    e_real, n_real = within_item_elasticity(panel(real), logp_week, tpw)
    res["price_response"] = {"real": {"elasticity": e_real, "item_weeks": n_real}}
    log(f"   real held-out baskets      elasticity {e_real:+.4f}  on {n_real:,} item-weeks")
    for sw in a.sweeps:
        e_g, n_g = within_item_elasticity(panel(gens[sw]), logp_week, tpw)
        res["price_response"][f"gibbs{sw}"] = {"elasticity": e_g, "item_weeks": n_g,
                                               "ratio_to_real": e_g / e_real if e_real else float("nan")}
        log(f"   generated, {sw} Gibbs sweeps   elasticity {e_g:+.4f}  on {n_g:,} item-weeks"
            f"   ({e_g / e_real:.0%} of real)" if e_real else "")

    tag = "" if a.label == "nested" else f"_{a.label}"
    with open(os.path.join(OUT, f"generator_eval{tag}.json"), "w") as f:
        json.dump(res, f, indent=2, default=float)

    # ------------------------------------------------------------------------ figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    best = f"gibbs{max(a.sweeps)}"
    fig, axes = plt.subplots(1, 4, figsize=(21, 4.6))
    for ax, (key, lab, hi) in zip(axes[:3], [("sizes", "items per basket", 30),
                                             ("cats", "categories per basket", 20),
                                             ("per_line", "units per line", 7)]):
        bins = np.arange(0, hi + 1)
        ax.hist(np.clip(S["real"][key], 0, hi), bins=bins, density=True, alpha=.6,
                color=PAL["grey"], label="real held-out")
        ax.hist(np.clip(S[best][key], 0, hi), bins=bins, density=True, alpha=.6,
                color=PAL["blue"], label="generated")
        ax.set_xlabel(lab)
        ax.set_ylabel("share of baskets" if key != "per_line" else "share of lines")
        ax.set_title(f"{lab}\nTVD {res['shape'][{'sizes': 'items per basket', 'cats': 'categories per basket', 'per_line': 'units per line'}[key]][best]['tvd_vs_real']:.3f}",
                     fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(alpha=.3)
    ax = axes[3]
    er = res["price_response"]["real"]["elasticity"]
    vals = [er] + [res["price_response"][f"gibbs{s}"]["elasticity"] for s in a.sweeps]
    labs = ["real"] + [f"gen, {s} sweeps" for s in a.sweeps]
    ax.bar(range(len(vals)), vals,
           color=[PAL["grey"]] + [PAL["blue"]] * len(a.sweeps))
    ax.axhline(0, color="k", lw=1)
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(labs, fontsize=8, rotation=15)
    ax.set_ylabel("within-item elasticity of buyers")
    ax.set_title("Price response recovered from\nthe generated data", fontsize=10)
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:+.3f}", ha="center", va="top", fontsize=8)
    ax.grid(axis="y", alpha=.3)
    fig.suptitle("Does the generated distribution match the real one?", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, f"generator_eval{tag}.png"), dpi=150, bbox_inches="tight")
    log("")
    log(f"wrote out/generator_eval{tag}.json and figures/generator_eval{tag}.png")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--label", default="nested")
    p.add_argument("--n-trips", type=int, default=6000)
    p.add_argument("--sweeps", type=int, nargs="+", default=[0, 4])
    p.add_argument("--top-items", type=int, default=300,
                   help="catalogue head used for the co-occurrence lift comparison")
    p.add_argument("--min-pair", type=int, default=5,
                   help="a pair needs this many joint occurrences across real+generated")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    main(p.parse_args())
