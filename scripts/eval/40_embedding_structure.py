"""
Stage 40 -- Does alpha agree with the sub-commodity hierarchy, and what does it look like?

Stage 39 established that alpha's nearest neighbours are 25x more likely than chance to
share a sub-commodity, but that ranking ALL pairs by cosine separates same-sub pairs only
weakly (AUC 0.583).  Those two facts are in tension, and this resolves it by asking the
question at every level of the hierarchy the data provides:

    department  (44)  ->  commodity/category  (188)  ->  sub-commodity  (758)

If alpha is organised coarsely, agreement should be high at department level and decay as
the labels get finer.  If it is organised finely, the reverse.  Measured four ways:

  1. k-NN purity at k = 1, 5, 10, 25, against the chance rate for each label set
  2. same-label AUC over all pairs, per level
  3. silhouette -- are the groups compact AND separated, or merely locally clustered
  4. k-means agreement at k = the number of true groups, by adjusted Rand and NMI

Then a t-SNE map coloured by department, with the sub-commodity structure marked, and a
table of nearest neighbours a human can check.

Writes out/embedding_structure_<label>.json and figures/embedding_tsne_<label>.png.
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
PAL = {"blue": "#2d6cdf", "grey": "#9aa5b1", "red": "#d1495b", "green": "#2a9d8f"}


def log(m):
    print(f"[40] {m}", flush=True)


def knn_purity(E, lab, ks, rng, n=2500):
    idx = rng.choice(len(E), size=min(n, len(E)), replace=False)
    S = E[idx] @ E[idx].T
    np.fill_diagonal(S, -np.inf)
    order = np.argsort(-S, axis=1)
    out = {}
    for k in ks:
        nn = order[:, :k]
        out[k] = float((lab[idx][nn] == lab[idx][:, None]).mean())
    # chance = probability two random members of the sample share a label
    _, cnt = np.unique(lab[idx], return_counts=True)
    out["chance"] = float(((cnt * (cnt - 1)).sum()) / (len(idx) * (len(idx) - 1)))
    return out


def same_label_auc(E, lab, rng, n=2000):
    idx = rng.choice(len(E), size=min(n, len(E)), replace=False)
    S = E[idx] @ E[idx].T
    iu = np.triu_indices(len(idx), 1)
    sc = S[iu]
    lb = (lab[idx][:, None] == lab[idx][None, :])[iu]
    if lb.sum() == 0 or (~lb).sum() == 0:
        return float("nan")
    o = np.argsort(-sc)
    lb = lb[o]
    tp = np.cumsum(lb) / lb.sum()
    fp = np.cumsum(~lb) / (~lb).sum()
    return float(np.trapezoid(tp, fp))


def main(a):
    os.makedirs(FIG, exist_ok=True)
    dev = torch.device("cpu")
    d = nb.NestedData(IN, device=dev)
    m, _ = cf.load(a.label, d, dev)
    A = m.alpha.detach().numpy()
    An = A / np.maximum(np.linalg.norm(A, axis=1, keepdims=True), 1e-9)

    items = pd.read_parquet(os.path.join(IN, "items.parquet")).sort_values("item_id")
    levels = {}
    for col, name in [("DEPARTMENT", "department"),
                      ("COMMODITY_DESC", "category"),
                      ("SUB_COMMODITY_DESC", "sub-commodity")]:
        if col in items.columns:
            levels[name] = pd.factorize(items[col])[0]
    rng = np.random.default_rng(a.seed)
    R = rng.normal(size=A.shape).astype(np.float32)
    Rn = R / np.linalg.norm(R, axis=1, keepdims=True)

    res = {"label": a.label, "levels": {}}
    log("")
    log("1-2. AGREEMENT WITH EACH LEVEL OF THE PRODUCT HIERARCHY")
    log(f"     {'level':14s} {'groups':>7s} {'chance':>8s} "
        f"{'1-NN':>7s} {'5-NN':>7s} {'10-NN':>7s} {'25-NN':>7s} {'AUC':>7s} {'AUC rnd':>8s}")
    for name, lab in levels.items():
        p = knn_purity(An, lab, [1, 5, 10, 25], np.random.default_rng(a.seed))
        auc = same_label_auc(An, lab, np.random.default_rng(a.seed))
        aucr = same_label_auc(Rn, lab, np.random.default_rng(a.seed))
        res["levels"][name] = {"n_groups": int(lab.max() + 1), "chance": p["chance"],
                               "knn": {str(k): p[k] for k in [1, 5, 10, 25]},
                               "auc": auc, "auc_random": aucr}
        log(f"     {name:14s} {lab.max()+1:7d} {p['chance']:8.4f} "
            f"{p[1]:7.4f} {p[5]:7.4f} {p[10]:7.4f} {p[25]:7.4f} {auc:7.4f} {aucr:8.4f}")
    log("     (purity = share of neighbours sharing the label; chance = random pair rate)")

    log("")
    log("3-4. GLOBAL GEOMETRY: are the groups compact and separated?")
    from sklearn.metrics import silhouette_score, adjusted_rand_score, \
        normalized_mutual_info_score
    from sklearn.cluster import MiniBatchKMeans
    sidx = rng.choice(len(An), size=min(3000, len(An)), replace=False)
    log(f"     {'level':14s} {'silhouette':>11s} {'ARI':>8s} {'NMI':>8s}")
    for name, lab in levels.items():
        sub = lab[sidx]
        keep = np.isin(sub, np.flatnonzero(np.bincount(sub) >= 2))
        sil = float(silhouette_score(An[sidx][keep], sub[keep], metric="cosine")) \
            if keep.sum() > 50 else float("nan")
        k = int(lab.max() + 1)
        km = MiniBatchKMeans(n_clusters=min(k, len(sidx) // 2), n_init=3,
                             random_state=a.seed).fit_predict(An[sidx])
        ari = float(adjusted_rand_score(sub, km))
        nmi = float(normalized_mutual_info_score(sub, km))
        res["levels"][name].update({"silhouette": sil, "ari": ari, "nmi": nmi})
        log(f"     {name:14s} {sil:11.4f} {ari:8.4f} {nmi:8.4f}")
    log("     (silhouette: +1 compact and separated, 0 touching, -1 wrong side.")
    log("      ARI/NMI: agreement of k-means at k = number of true groups; 0 = chance)")

    # ------------------------------------------------------------------ examples
    log("")
    log("5. NEAREST NEIGHBOURS, for inspection")
    nm = items.SUB_COMMODITY_DESC.to_numpy() if "SUB_COMMODITY_DESC" in items else None
    cnt = np.bincount(d.splits["train"]["item"], minlength=d.J)
    pop = np.argsort(-cnt)[:a.n_examples * 12][::12][:a.n_examples]
    ex = []
    for j in pop:
        s = An @ An[j]
        s[j] = -np.inf
        nn = np.argsort(-s)[:3]
        ex.append({"item": str(nm[j]), "neighbours": [str(nm[q]) for q in nn],
                   "cos": [float(s[q]) for q in nn]})
        log(f"     {str(nm[j])[:34]:34s} -> " +
            " | ".join(f"{str(nm[q])[:26]} ({s[q]:.2f})" for q in nn))
    res["examples"] = ex

    # ------------------------------------------------------------------ t-SNE
    log("")
    log("6. t-SNE map")
    from sklearn.manifold import TSNE
    tidx = rng.choice(len(An), size=min(a.n_tsne, len(An)), replace=False)
    Z = TSNE(n_components=2, metric="cosine", init="pca", perplexity=30,
             random_state=a.seed).fit_transform(An[tidx])
    dep = levels.get("department", np.zeros(len(An), dtype=int))[tidx]
    sub = levels.get("sub-commodity", np.zeros(len(An), dtype=int))[tidx]
    dname = items.DEPARTMENT.to_numpy()[tidx] if "DEPARTMENT" in items else None

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(14, 6.4))
    top = pd.Series(dname).value_counts().head(10).index.tolist()
    cmap = plt.get_cmap("tab10")
    ax[0].scatter(Z[:, 0], Z[:, 1], s=4, c="#d8dde2", linewidths=0)
    for i, dp in enumerate(top):
        msk = dname == dp
        ax[0].scatter(Z[msk, 0], Z[msk, 1], s=6, color=cmap(i), linewidths=0,
                      label=f"{dp} ({msk.sum()})")
    ax[0].legend(fontsize=6.5, markerscale=2, loc="upper right", framealpha=.9)
    ax[0].set_title("coloured by department — coarse structure", fontsize=10)

    # right panel: the largest sub-commodities, to show whether FINE labels cohere
    tops = pd.Series(sub).value_counts().head(12).index.tolist()
    ax[1].scatter(Z[:, 0], Z[:, 1], s=4, c="#d8dde2", linewidths=0)
    subname = items.SUB_COMMODITY_DESC.to_numpy()[tidx]
    for i, sc in enumerate(tops):
        msk = sub == sc
        ax[1].scatter(Z[msk, 0], Z[msk, 1], s=14, color=cmap(i % 10), linewidths=0,
                      label=f"{str(subname[msk][0])[:24]} ({msk.sum()})")
    ax[1].legend(fontsize=6.5, markerscale=1.5, loc="upper right", framealpha=.9)
    ax[1].set_title("the 12 largest sub-commodities — fine structure", fontsize=10)
    for x in ax:
        x.set_xticks([]); x.set_yticks([])
    fig.suptitle(f"t-SNE of alpha, cosine metric, {len(tidx):,} products ({a.label})",
                 fontsize=11)
    fig.tight_layout()
    fp = os.path.join(FIG, f"embedding_tsne_{a.label}.png")
    fig.savefig(fp, dpi=150, bbox_inches="tight")
    log(f"   wrote {fp}")

    with open(os.path.join(OUT, f"embedding_structure_{a.label}.json"), "w") as f:
        json.dump(res, f, indent=2)
    log(f"   wrote out/embedding_structure_{a.label}.json")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--label", default="spec_nested")
    p.add_argument("--n-tsne", type=int, default=3000)
    p.add_argument("--n-examples", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    main(p.parse_args())
