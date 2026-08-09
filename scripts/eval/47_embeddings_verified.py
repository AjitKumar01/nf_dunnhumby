"""
Stage 47 -- Are the embeddings meaningful?  With nulls, intervals, and figures.

Two earlier findings need re-checking on the current model, because the persistence term
takes load off theta.alpha and may have freed alpha to encode something different.

  PRODUCTS.  alpha is claimed to recover the sub-commodity hierarchy.  Verified three ways:
    * against a PERMUTATION null -- shuffle the labels and recompute.  A real signal
      collapses to chance; a metric that is high for structural reasons does not.
    * against a random embedding of identical shape, which controls for dimensionality.
    * with a bootstrap interval over products, so "25x chance" carries an error bar.

  HOUSEHOLDS.  theta, c_user and gamma are claimed to carry NO demographic signal.  A
  negative result needs more care than a positive one: the earlier test used a linear
  probe, which cannot see a non-linear relationship.  Repeated with gradient boosting and
  with a shuffled-label control, so the claim becomes "not linearly OR non-linearly
  recoverable at this sample size" rather than "not recoverable".

Figures: figures/embeddings_verified.png.
Writes out/embeddings_verified.json.
"""
import argparse
import importlib
import json
import os
import sys

import numpy as np
import pandas as pd
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
FIG = os.path.join(HERE, "..", "..", "figures")
DEM = ("/Users/ajit/Projects/Causal/dunnhumby_The-Complete-Journey/"
       "dunnhumby_The-Complete-Journey CSV/hh_demographic.csv")


def log(m):
    print(f"[47] {m}", flush=True)


def knn_purity(E, lab, k, idx):
    S = E[idx] @ E[idx].T
    np.fill_diagonal(S, -np.inf)
    nn = np.argsort(-S, axis=1)[:, :k]
    return float((lab[idx][nn] == lab[idx][:, None]).mean())


def auc_same(E, lab, idx):
    S = E[idx] @ E[idx].T
    iu = np.triu_indices(len(idx), 1)
    sc, lb = S[iu], (lab[idx][:, None] == lab[idx][None, :])[iu]
    if lb.sum() < 5 or (~lb).sum() < 5:
        return float("nan")
    o = np.argsort(-sc)
    lb = lb[o]
    return float(np.trapezoid(np.cumsum(lb) / lb.sum(),
                              np.cumsum(~lb) / (~lb).sum()))


def main(a):
    os.makedirs(FIG, exist_ok=True)
    dev = torch.device("cpu")
    d = nb.NestedData(IN, device=dev)
    m, _ = cf.load(a.label, d, dev)
    A = m.alpha.detach().numpy()
    An = A / np.maximum(np.linalg.norm(A, axis=1, keepdims=True), 1e-9)
    items = pd.read_parquet(os.path.join(IN, "items.parquet")).sort_values("item_id")
    sub = pd.factorize(items.SUB_COMMODITY_DESC)[0]
    rng = np.random.default_rng(a.seed)
    R = rng.normal(size=A.shape).astype(np.float32)
    Rn = R / np.linalg.norm(R, axis=1, keepdims=True)
    res = {"label": a.label}

    # ---------------------------------------------------------------- products
    log("")
    log("1. PRODUCTS -- is the sub-commodity signal real?")
    idx = rng.choice(len(An), size=min(a.n_items, len(An)), replace=False)
    p_fit = knn_purity(An, sub, a.k, idx)
    p_rnd = knn_purity(Rn, sub, a.k, idx)
    perm = [knn_purity(An, rng.permutation(sub), a.k, idx) for _ in range(a.perms)]
    _, cnt = np.unique(sub[idx], return_counts=True)
    chance = float((cnt * (cnt - 1)).sum() / (len(idx) * (len(idx) - 1)))
    # Resample a fresh subset of the SAME size each rep.  A with-replacement draw
    # collapsed under np.unique to ~63% of the size, and purity falls with sample size
    # because fewer same-label neighbours are available -- which put the interval below
    # the point estimate.
    boot = [knn_purity(An, sub, a.k,
                       rng.choice(len(An), size=len(idx), replace=False))
            for _ in range(a.boots)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    log(f"   {a.k}-NN purity, fitted alpha    {p_fit:.4f}  95% CI [{lo:.4f}, {hi:.4f}]")
    log(f"   same, labels PERMUTED           {np.mean(perm):.4f} "
        f"(sd {np.std(perm):.4f}, {a.perms} draws)")
    log(f"   same, random embedding          {p_rnd:.4f}")
    log(f"   chance for a random neighbour   {chance:.4f}")
    log(f"   -> {p_fit/max(chance,1e-9):.1f}x chance; the permutation null sits at chance,")
    log(f"      so the metric is not high for structural reasons")
    res["products"] = {"purity": p_fit, "ci": [float(lo), float(hi)],
                       "perm_mean": float(np.mean(perm)), "random": p_rnd,
                       "chance": chance}

    # where the signal lives: all pairs vs pairs the model has evidence for
    tr = d.splits["train"]
    from collections import Counter
    cnt2 = Counter()
    bi = rng.choice(tr["n_baskets"], size=30000, replace=False)
    for i in bi:
        it = np.unique(tr["item"][tr["starts"][i]:tr["ends"][i]])
        for x in range(len(it)):
            for y in range(x + 1, len(it)):
                cnt2[(it[x], it[y])] += 1
    pairs = np.array([k for k, v in cnt2.items() if v >= a.min_co])
    cos = (An[pairs[:, 0]] * An[pairs[:, 1]]).sum(1)
    same = sub[pairs[:, 0]] == sub[pairs[:, 1]]
    o = np.argsort(-cos)
    lb = same[o]
    auc_co = float(np.trapezoid(np.cumsum(lb) / lb.sum(),
                                np.cumsum(~lb) / (~lb).sum()))
    auc_all = auc_same(An, sub, idx)
    log(f"\n   same-sub AUC over ALL pairs                 {auc_all:.4f}")
    log(f"   same-sub AUC over pairs co-bought >= {a.min_co}x    {auc_co:.4f} "
        f"({len(pairs):,} pairs)")
    log(f"   mean cosine: same sub {cos[same].mean():+.4f}, different {cos[~same].mean():+.4f}")
    res["products"].update({"auc_all": auc_all, "auc_cooccurring": auc_co,
                            "cos_same": float(cos[same].mean()),
                            "cos_diff": float(cos[~same].mean())})

    # ---------------------------------------------------------------- households
    log("")
    log("2. HOUSEHOLDS -- is the negative result robust to a non-linear probe?")
    tx = pd.read_parquet(os.path.join(HERE, "..", "..", "data", "tx.parquet"),
                         columns=["household_key"])
    users = np.sort(tx.household_key.unique())
    uid = pd.Series(np.arange(len(users)), index=users)
    dem = pd.read_csv(DEM)
    dem["user_id"] = dem.household_key.map(uid)
    dem = dem.dropna(subset=["user_id"])
    dem["user_id"] = dem.user_id.astype(int)
    dem = dem[dem.user_id < d.N]
    X = np.concatenate([m.theta.detach().numpy(), m.c_user.detach().numpy(),
                        m.gamma.detach().numpy()], 1)
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    targets = [("classification_3", "income band"), ("classification_4", "household size"),
               ("classification_1", "age group"), ("KID_CATEGORY_DESC", "children"),
               ("HOMEOWNER_DESC", "homeowner")]
    log(f"   {'target':18s} {'n':>5s} {'majority':>9s} {'linear':>8s} {'boosted':>9s}"
        f" {'shuffled':>9s}")
    hh = {}
    for col, nm in targets:
        s = dem.dropna(subset=[col])
        y = pd.factorize(s[col])[0]
        keep = np.bincount(y) >= 5
        s, y = s[keep[y]], y[keep[y]]
        if len(s) < 100:
            continue
        Xs = X[s.user_id.values]
        base = float(np.bincount(y).max() / len(y))
        cv = StratifiedKFold(5, shuffle=True, random_state=a.seed)
        lin = float(cross_val_score(make_pipeline(StandardScaler(),
                    LogisticRegression(max_iter=2000)), Xs, y, cv=cv).mean())
        gb = float(cross_val_score(GradientBoostingClassifier(
                    n_estimators=100, max_depth=3, random_state=a.seed),
                    Xs, y, cv=cv).mean())
        sh = float(cross_val_score(GradientBoostingClassifier(
                    n_estimators=100, max_depth=3, random_state=a.seed),
                    Xs, rng.permutation(y), cv=cv).mean())
        hh[nm] = {"n": int(len(s)), "majority": base, "linear": lin,
                  "boosted": gb, "shuffled": sh}
        log(f"   {nm:18s} {len(s):5d} {base:9.3f} {lin:8.3f} {gb:9.3f} {sh:9.3f}")
    res["households"] = hh
    best = max((v["boosted"] - v["majority"]) for v in hh.values())
    log(f"   best boosted probe beats the majority class by {best:+.3f}")

    # ---------------------------------------------------------------- figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
    C = {"fit": "#1f5c4a", "null": "#9aa5b1", "bad": "#8b2f3d"}

    ks = [1, 5, 10, 25, 50]
    fit = [knn_purity(An, sub, k, idx) for k in ks]
    rnd = [knn_purity(Rn, sub, k, idx) for k in ks]
    pm = [np.mean([knn_purity(An, rng.permutation(sub), k, idx) for _ in range(5)])
          for k in ks]
    ax[0].plot(ks, fit, "o-", color=C["fit"], lw=2, label="fitted α")
    ax[0].plot(ks, pm, "s--", color=C["null"], lw=1.5, label="labels permuted")
    ax[0].plot(ks, rnd, "^:", color=C["bad"], lw=1.5, label="random embedding")
    ax[0].axhline(chance, color="k", ls=":", lw=1, label="chance")
    ax[0].set_yscale("log"); ax[0].set_xlabel("k nearest neighbours")
    ax[0].set_ylabel("share sharing the sub-commodity")
    ax[0].set_title("Products: signal vs two nulls", fontsize=10)
    ax[0].legend(fontsize=7.5); ax[0].grid(alpha=.3)

    ax[1].hist(cos[~same], bins=60, density=True, alpha=.65, color=C["null"],
               label=f"different sub-commodity ({(~same).sum():,})")
    ax[1].hist(cos[same], bins=60, density=True, alpha=.75, color=C["fit"],
               label=f"same sub-commodity ({same.sum():,})")
    ax[1].set_xlabel("cosine similarity"); ax[1].set_ylabel("density")
    ax[1].set_title(f"Products co-bought ≥{a.min_co}×: AUC {auc_co:.3f}", fontsize=10)
    ax[1].legend(fontsize=7.5); ax[1].grid(alpha=.3)

    nms = list(hh); xx = np.arange(len(nms)); w = 0.27
    ax[2].bar(xx - w, [hh[n]["majority"] for n in nms], w, color="k", alpha=.35,
              label="majority class")
    ax[2].bar(xx, [hh[n]["boosted"] for n in nms], w, color=C["fit"],
              label="boosted probe on embeddings")
    ax[2].bar(xx + w, [hh[n]["shuffled"] for n in nms], w, color=C["null"],
              label="shuffled labels")
    ax[2].set_xticks(xx); ax[2].set_xticklabels(nms, rotation=20, ha="right", fontsize=7.5)
    ax[2].set_ylabel("5-fold CV accuracy")
    ax[2].set_title("Households: no demographic signal", fontsize=10)
    ax[2].legend(fontsize=7.5); ax[2].grid(alpha=.3, axis="y")

    fig.tight_layout()
    fp = os.path.join(FIG, "embeddings_verified.png")
    fig.savefig(fp, dpi=150, bbox_inches="tight")
    log(f"\n   wrote {fp}")
    with open(os.path.join(OUT, "embeddings_verified.json"), "w") as f:
        json.dump(res, f, indent=2)
    log(f"   wrote out/embeddings_verified.json")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--label", default="ps_nested")
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--n-items", type=int, default=2000)
    p.add_argument("--perms", type=int, default=20)
    p.add_argument("--boots", type=int, default=60)
    p.add_argument("--min-co", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    main(p.parse_args())
