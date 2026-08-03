"""
Stage 24 -- Do the learned embeddings recover the sub-commodity structure?

The requirement is specific: after fitting, products should reveal sub-commodity
level clusters.  The model is never shown SUB_COMMODITY_DESC -- not as a feature, not
as a grouping, not in the likelihood -- so this is a clean test of whether the
embedding learned real product structure or just memorised popularity.

Five measurements, from weakest to strongest:

  1. nearest-neighbour purity   for each item, the share of its k nearest neighbours
                                (cosine on alpha) sharing its sub-commodity, against
                                the rate a random neighbour would achieve
  2. same-sub AUC               rank every item pair by cosine similarity and ask how
                                well that ranking separates same-sub-commodity pairs
                                from the rest.  0.5 is nothing, 1.0 is perfect
  3. silhouette                 do sub-commodities form compact, separated groups
  4. clustering agreement       k-means at k = number of sub-commodities, scored by
                                adjusted Rand and normalised mutual information
  5. a 2-D map                  t-SNE coloured by department, plus nearest-neighbour
                                examples that a human can check

Every measurement is run against controls, because a number alone proves nothing:

  * a random embedding of the same shape        -- the floor
  * a popularity-only embedding                 -- rules out "it just learned volume"
  * the paper's model (nf), on its own 560 items -- the head-to-head that matters
  * the basket model restricted to those same 560 items, so the comparison with nf is
    on identical items and identical ground truth rather than two different universes

Writes out/embedding_eval.json, out/embedding_neighbours_basket.csv and
figures/embedding_*.png.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
IN = os.path.join(HERE, "..", "basket_input")
DATA = os.path.join(HERE, "..", "data")
OUT = os.path.join(HERE, "..", "out")
FIG = os.path.join(HERE, "..", "figures")
MI = os.path.join(HERE, "..", "model_input")

PALETTE = {"blue": "#2d6cdf", "grey": "#9aa5b1", "red": "#d1495b",
           "green": "#2a9d8f", "amber": "#e9c46a", "purple": "#7b6cd9"}


def log(m):
    print(f"[24] {m}", flush=True)


def unit(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(n, 1e-9, None)


def knn_purity(E, labels, k=10, block=512):
    """Share of each item's k nearest neighbours that share its label."""
    U = unit(E)
    n = len(U)
    k = min(k, n - 1)
    hits = np.zeros(n)
    for a in range(0, n, block):
        sim = U[a:a + block] @ U.T
        for r in range(sim.shape[0]):
            sim[r, a + r] = -np.inf              # never count the item itself
        idx = np.argpartition(-sim, k, axis=1)[:, :k]
        hits[a:a + block] = (labels[idx] == labels[a:a + block, None]).mean(1)
    return hits


def same_label_auc(E, labels, max_items=3000, seed=0):
    """AUC of cosine similarity for predicting 'same sub-commodity'.

    Computed on a subsample of pairs: the full pair set is O(J^2) and the estimate is
    already tight at a few million pairs.
    """
    rng = np.random.default_rng(seed)
    n = len(E)
    if n > max_items:
        sel = rng.choice(n, max_items, replace=False)
        E, labels = E[sel], labels[sel]
    U = unit(E)
    S = U @ U.T
    iu = np.triu_indices(len(U), k=1)
    sim = S[iu]
    same = (labels[iu[0]] == labels[iu[1]])
    if same.sum() == 0 or (~same).sum() == 0:
        return float("nan")
    # rank-based AUC (Mann-Whitney)
    order = np.argsort(sim)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(sim) + 1)
    n1, n0 = same.sum(), (~same).sum()
    return float((ranks[same].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def evaluate_embedding(name, E, labels, dept, k=10, seed=0):
    from sklearn.cluster import KMeans
    from sklearn.metrics import (silhouette_score, adjusted_rand_score,
                                 normalized_mutual_info_score)
    r = {"name": name, "n_items": int(len(E)), "dim": int(E.shape[1]),
         "n_labels": int(len(np.unique(labels)))}
    pur = knn_purity(E, labels, k=k)
    # Chance purity is not 1/n_labels: sub-commodities differ hugely in size, and a
    # random neighbour lands in one with probability proportional to its size.
    _, cnt = np.unique(labels, return_counts=True)
    chance = float(((cnt / cnt.sum()) ** 2).sum())
    r["knn_purity"] = float(pur.mean())
    r["knn_purity_chance"] = chance
    r["knn_purity_lift"] = float(pur.mean() / chance) if chance else float("nan")
    r["share_items_with_a_same_sub_neighbour"] = float((pur > 0).mean())
    r["same_sub_auc"] = same_label_auc(E, labels, seed=seed)
    U = unit(E)
    try:
        sel = np.random.default_rng(seed).choice(
            len(U), min(3000, len(U)), replace=False)
        r["silhouette_sub"] = float(silhouette_score(U[sel], labels[sel], metric="cosine"))
    except Exception:
        r["silhouette_sub"] = float("nan")
    km = KMeans(n_clusters=min(r["n_labels"], len(U) - 1), n_init=4,
                random_state=seed).fit(U)
    r["kmeans_ari_vs_sub"] = float(adjusted_rand_score(labels, km.labels_))
    r["kmeans_nmi_vs_sub"] = float(normalized_mutual_info_score(labels, km.labels_))
    if dept is not None:
        r["knn_purity_department"] = float(knn_purity(E, dept, k=k).mean())
    return r


def main(a):
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(FIG, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    items = pd.read_parquet(os.path.join(IN, "items.parquet"))
    meta = json.load(open(os.path.join(IN, "meta.json")))
    J = int(meta["n_items"])
    labels = items.sort_values("item_id").sub_id.to_numpy()
    dept = items.sort_values("item_id").dept_id.to_numpy()
    pid = items.sort_values("item_id").PRODUCT_ID.to_numpy()
    subname = items.sort_values("item_id").SUB_COMMODITY_DESC.to_numpy()
    log(f"{J:,} items, {len(np.unique(labels)):,} sub-commodities, "
        f"{len(np.unique(dept))} departments")

    results, embeddings = [], {}
    rng = np.random.default_rng(a.seed)

    # ---------------------------------------------------------- basket models
    for lb in a.labels:
        path = os.path.join(OUT, f"{lb}{a.suffix}.pt")
        if not os.path.exists(path):
            log(f"  {lb}: no checkpoint, skipping")
            continue
        sd = torch.load(path, map_location="cpu")
        E = sd["alpha"].numpy()
        embeddings[lb] = E
        r = evaluate_embedding(lb, E, labels, dept, k=a.k, seed=a.seed)
        results.append(r)
        log(f"  {lb:18s} kNN purity {r['knn_purity']:.4f} "
            f"({r['knn_purity_lift']:.1f}x chance)   AUC {r['same_sub_auc']:.4f}   "
            f"silhouette {r['silhouette_sub']:+.4f}   NMI {r['kmeans_nmi_vs_sub']:.4f}")

    # ------------------------------------------------------------- controls
    primary = a.primary if a.primary in embeddings else (
        a.labels[0] if a.labels[0] in embeddings else None)
    base = embeddings.get(primary)
    if primary:
        log(f"headline model for the head-to-head, neighbours and map: {primary}")
    if base is not None:
        R = rng.normal(size=base.shape).astype(np.float32)
        r = evaluate_embedding("control: random", R, labels, dept, k=a.k, seed=a.seed)
        results.append(r)
        log(f"  {'control: random':18s} kNN purity {r['knn_purity']:.4f} "
            f"({r['knn_purity_lift']:.1f}x chance)   AUC {r['same_sub_auc']:.4f}")

        # Popularity-only control: one informative dimension, so any structure it
        # shows is what volume alone can produce.
        cnt = items.sort_values("item_id").n_lines.to_numpy(dtype=np.float64)
        P = np.zeros_like(base)
        P[:, 0] = np.log(cnt)
        P += rng.normal(scale=1e-3, size=P.shape)
        r = evaluate_embedding("control: popularity", P, labels, dept, k=a.k, seed=a.seed)
        results.append(r)
        log(f"  {'control: popularity':18s} kNN purity {r['knn_purity']:.4f} "
            f"({r['knn_purity_lift']:.1f}x chance)   AUC {r['same_sub_auc']:.4f}")

    # ------------------------------------- head-to-head with the paper's model
    # nf models 560 items in 56 categories.  Comparing its embedding on those items
    # against the basket embedding on 5,455 items would confound the model with the
    # item universe, so both are measured on the same 560 items.
    nf_path = os.path.join(OUT, "nf_stage1.pt")
    map_path = os.path.join(MI, "id_maps", "items.csv")
    if os.path.exists(nf_path) and os.path.exists(map_path):
        nf_map = pd.read_csv(map_path)
        sd = torch.load(nf_path, map_location="cpu")
        beta = sd["beta.mu"].numpy() if "beta.mu" in sd else None
        if beta is not None:
            nf_map = nf_map.sort_values("item_id")
            nf_pid = nf_map.PRODUCT_ID.to_numpy()
            sub_of = items.set_index("PRODUCT_ID").sub_id
            keep = np.array([p in sub_of.index for p in nf_pid])
            nf_lab = np.array([sub_of.get(p, -1) for p in nf_pid])
            # only items whose sub-commodity has a second member here, otherwise the
            # question "does it neighbour its own kind" has no answer
            _, c = np.unique(nf_lab[keep], return_counts=True)
            ok_lab = set(np.unique(nf_lab[keep])[c >= 2])
            sel = keep & np.array([l in ok_lab for l in nf_lab])
            log(f"head-to-head on {int(sel.sum())} items that nf models and whose "
                f"sub-commodity has 2+ members")
            if sel.sum() > 30:
                r = evaluate_embedding("nf (paper model), 560-item universe",
                                       beta[sel], nf_lab[sel], None, k=a.k, seed=a.seed)
                results.append(r)
                log(f"  {'nf beta':18s} kNN purity {r['knn_purity']:.4f} "
                    f"({r['knn_purity_lift']:.1f}x chance)   AUC {r['same_sub_auc']:.4f}   "
                    f"silhouette {r['silhouette_sub']:+.4f}")
                if base is not None:
                    pid_to_row = {p: i for i, p in enumerate(pid)}
                    rows = np.array([pid_to_row[p] for p in nf_pid[sel]])
                    r = evaluate_embedding(f"{primary}, same 560-item universe",
                                           base[rows], nf_lab[sel], None, k=a.k,
                                           seed=a.seed)
                    results.append(r)
                    log(f"  {primary + ' alpha':18s} kNN purity {r['knn_purity']:.4f} "
                        f"({r['knn_purity_lift']:.1f}x chance)   AUC {r['same_sub_auc']:.4f}   "
                        f"silhouette {r['silhouette_sub']:+.4f}")

    with open(os.path.join(OUT, "embedding_eval.json"), "w") as f:
        json.dump(results, f, indent=2)

    # ------------------------------------------------ qualitative neighbours
    if base is not None:
        U = unit(base)
        rng2 = np.random.default_rng(7)
        big = np.argsort(-items.sort_values("item_id").n_lines.to_numpy())[:400]
        picks = rng2.choice(big, 15, replace=False)
        rows = []
        for i in picks:
            sim = U @ U[i]
            sim[i] = -np.inf
            nb = np.argsort(-sim)[:5]
            for rank, j in enumerate(nb, 1):
                rows.append({"query_product": pid[i], "query_sub": subname[i],
                             "rank": rank, "neighbour_product": pid[j],
                             "neighbour_sub": subname[j],
                             "same_sub": int(labels[i] == labels[j]),
                             "cosine": float(sim[j])})
        NB = pd.DataFrame(rows)
        NB.to_csv(os.path.join(OUT, "embedding_neighbours_basket.csv"), index=False)
        log(f"nearest-neighbour examples: {NB.same_sub.mean():.1%} of the top-5 "
            f"neighbours of 15 popular items share the query's sub-commodity")

    # ------------------------------------------------------------- figures
    R = pd.DataFrame(results)
    if len(R):
        fig, axes = plt.subplots(1, 3, figsize=(17, 5.0))
        order = R.sort_values("knn_purity")
        ax = axes[0]
        cols = [PALETTE["grey"] if "control" in n else
                (PALETTE["red"] if n.startswith("nf") else PALETTE["blue"])
                for n in order.name]
        ax.barh(range(len(order)), order.knn_purity, color=cols)
        ax.plot(order.knn_purity_chance, range(len(order)), "k|", ms=14,
                label="chance for that item set")
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels(order.name, fontsize=8)
        ax.set_xlabel(f"share of the {a.k} nearest neighbours in the same sub-commodity")
        ax.set_title("Nearest-neighbour purity\nthe model is never shown sub-commodity",
                     fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(axis="x", alpha=.3)
        for i, v in enumerate(order.knn_purity):
            ax.text(v + .005, i, f"{v:.3f}", va="center", fontsize=8)

        ax = axes[1]
        o2 = R.sort_values("same_sub_auc")
        cols = [PALETTE["grey"] if "control" in n else
                (PALETTE["red"] if n.startswith("nf") else PALETTE["blue"])
                for n in o2.name]
        ax.barh(range(len(o2)), o2.same_sub_auc, color=cols)
        ax.axvline(.5, color="k", ls="--", lw=1, label="no information")
        ax.set_yticks(range(len(o2)))
        ax.set_yticklabels(o2.name, fontsize=8)
        ax.set_xlim(.4, 1.0)
        ax.set_xlabel("AUC: does cosine similarity identify same-sub-commodity pairs?")
        ax.set_title("Pairwise separation", fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(axis="x", alpha=.3)
        for i, v in enumerate(o2.same_sub_auc):
            ax.text(v + .005, i, f"{v:.3f}", va="center", fontsize=8)

        ax = axes[2]
        o3 = R.dropna(subset=["silhouette_sub"]).sort_values("silhouette_sub")
        cols = [PALETTE["grey"] if "control" in n else
                (PALETTE["red"] if n.startswith("nf") else PALETTE["blue"])
                for n in o3.name]
        ax.barh(range(len(o3)), o3.silhouette_sub, color=cols)
        ax.axvline(0, color="k", lw=1)
        ax.set_yticks(range(len(o3)))
        ax.set_yticklabels(o3.name, fontsize=8)
        ax.set_xlabel("silhouette (cosine) with sub-commodity as the label")
        ax.set_title("Cluster compactness", fontsize=10)
        ax.grid(axis="x", alpha=.3)
        fig.suptitle("Do the learned product embeddings recover sub-commodity structure?",
                     fontsize=12)
        fig.tight_layout()
        fig.savefig(os.path.join(FIG, "embedding_scores.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

    # --- t-SNE map
    if base is not None and a.tsne:
        from sklearn.manifold import TSNE
        sel = np.argsort(-items.sort_values("item_id").n_lines.to_numpy())[:a.tsne_items]
        log(f"t-SNE on the {len(sel)} most-purchased items ...")
        Z = TSNE(n_components=2, perplexity=30, init="pca", random_state=a.seed,
                 max_iter=1000).fit_transform(unit(base)[sel])
        dep = dept[sel]
        dnames = (items.sort_values("item_id").DEPARTMENT.to_numpy())[sel]
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        ax = axes[0]
        top_dep = pd.Series(dnames).value_counts().head(10).index
        cmap = plt.get_cmap("tab10")
        for i, dn in enumerate(top_dep):
            m = dnames == dn
            ax.scatter(Z[m, 0], Z[m, 1], s=9, alpha=.75, color=cmap(i), label=str(dn)[:22])
        m = ~np.isin(dnames, top_dep)
        ax.scatter(Z[m, 0], Z[m, 1], s=6, alpha=.25, color="#cccccc", label="other")
        ax.legend(fontsize=7, markerscale=2, loc="best")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title("Coloured by DEPARTMENT (never shown to the model)", fontsize=10)

        ax = axes[1]
        big_subs = pd.Series(subname[sel]).value_counts().head(12).index
        for i, sn in enumerate(big_subs):
            m = subname[sel] == sn
            ax.scatter(Z[m, 0], Z[m, 1], s=16, alpha=.85, color=cmap(i % 10),
                       label=str(sn)[:26])
        m = ~np.isin(subname[sel], big_subs)
        ax.scatter(Z[m, 0], Z[m, 1], s=5, alpha=.15, color="#cccccc")
        ax.legend(fontsize=7, markerscale=2, loc="best")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title("The 12 largest SUB-COMMODITIES", fontsize=10)
        fig.suptitle("Item embedding, t-SNE: structure the model was never given",
                     fontsize=12)
        fig.tight_layout()
        fig.savefig(os.path.join(FIG, "embedding_tsne.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

    log("wrote out/embedding_eval.json and figures/embedding_*.png")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--labels", nargs="+",
                   default=["basket", "basket_noctx", "basket_nostate",
                            "basket_noprice", "basket_pop"])
    p.add_argument("--primary", default="tied_k64_r",
                   help="model whose embedding is used for the head-to-head "
                        "with nf, the neighbour examples and the t-SNE map")
    p.add_argument("--suffix", default="_basket",
                   help="checkpoint suffix: _basket for stage 23, _nested "
                        "for stage 27 (both store the embedding as alpha)")
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--tsne", action="store_true", default=True)
    p.add_argument("--tsne-items", type=int, default=2500)
    p.add_argument("--seed", type=int, default=0)
    main(p.parse_args())
