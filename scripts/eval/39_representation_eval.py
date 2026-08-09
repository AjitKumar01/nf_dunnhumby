"""
Stage 39 -- What did the embeddings and the price coefficients actually learn?

Three questions the fitted model has never been asked.

  1. HOUSEHOLD EMBEDDINGS.  theta_i, c^u_i, gamma_i and h_i are fitted from purchases
     alone.  dunnhumby ships household demographics -- income, size, children, age --
     that the model NEVER sees.  If a linear probe on the embedding predicts them above
     chance, the embedding has recovered something real about who the household is.
     Scored by 5-fold cross-validation against two floors: a shuffled-label control
     (the same probe on permuted targets) and a purchase-count-only baseline (rules out
     "it just learned how much they shop").

  2. PRODUCT EMBEDDINGS.  alpha_j carries both taste (theta.alpha in Eq. 4) and
     co-purchase (alpha.alpha in Eq. 11), so it is worth knowing which structure it
     encodes.  Nearest-neighbour purity and same-sub-commodity AUC against a random
     embedding of identical shape.

  3. PRICE SENSITIVITY.  gamma_i.beta_j is a household x product matrix delivered as a
     rank-K_p bilinear form.  Its sign is unconstrained, so the model can express
     upward-sloping demand; how often does it?  How much of the variation is between
     households versus between products?  Does household price sensitivity line up with
     income, which the model never saw?  And does the choice margin (gamma.beta) agree
     with the quantity margin (gamma^q.beta^q), which is fitted separately?

Writes out/representation_eval_<label>.json.
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
DATA = os.path.join(HERE, "..", "..", "data")
OUT = os.path.join(HERE, "..", "..", "out")
DEM = ("/Users/ajit/Projects/Causal/dunnhumby_The-Complete-Journey/"
       "dunnhumby_The-Complete-Journey CSV/hh_demographic.csv")


def log(m):
    print(f"[39] {m}", flush=True)


def cv_probe(X, y, classes, seed=0, folds=5):
    """Ridge (regression) or multinomial logistic (classification), 5-fold CV.

    Returns the out-of-fold score: R^2 for continuous targets, accuracy for discrete.
    Standardises inside each fold so no test information leaks through the scaling.
    """
    from sklearn.linear_model import RidgeCV, LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
    if classes:
        mdl = make_pipeline(StandardScaler(),
                            LogisticRegression(max_iter=2000, C=1.0))
        cv = StratifiedKFold(folds, shuffle=True, random_state=seed)
        return float(cross_val_score(mdl, X, y, cv=cv, scoring="accuracy").mean())
    mdl = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 4, 13)))
    cv = KFold(folds, shuffle=True, random_state=seed)
    return float(cross_val_score(mdl, X, y, cv=cv, scoring="r2").mean())


def main(a):
    dev = torch.device("cpu")
    d = nb.NestedData(IN, device=dev)
    m, cfg = cf.load(a.label, d, dev)
    res = {"label": a.label}

    # ---------------------------------------------------------------- households
    log("")
    log("1. HOUSEHOLD EMBEDDINGS vs demographics the model never saw")
    tx = pd.read_parquet(os.path.join(DATA, "tx.parquet"),
                         columns=["household_key"])
    users = np.sort(tx.household_key.unique())
    uid = pd.Series(np.arange(len(users)), index=users)      # same map as 22_basket_data
    dem = pd.read_csv(DEM)
    dem["user_id"] = dem.household_key.map(uid)
    dem = dem.dropna(subset=["user_id"])
    dem["user_id"] = dem.user_id.astype(int)
    dem = dem[dem.user_id < d.N]
    log(f"   {len(dem):,} of {d.N:,} households have demographics "
        f"({len(dem)/d.N:.0%})")

    theta = m.theta.detach().numpy()
    blocks = {"theta (taste)": theta}
    if getattr(m, "c_user", None) is not None:
        blocks["c_user (category appetite)"] = m.c_user.detach().numpy()
    blocks["gamma (price)"] = m.gamma.detach().numpy()
    blocks["all concatenated"] = np.concatenate(
        [v for v in blocks.values()], axis=1)

    tr = d.splits["train"]
    n_trips = np.bincount(tr["user"], minlength=d.N).astype(float)
    n_items = np.zeros(d.N)
    np.add.at(n_items, tr["user"], 1.0)
    volume = np.stack([np.log1p(n_trips), np.log1p(n_items)], 1)
    blocks["volume only (control)"] = volume

    # This copy of hh_demographic.csv has anonymised headers.  Mapped by their value
    # sets: classification_3 is Level1..Level12 (the 12 dunnhumby income bands),
    # classification_4 is 1..5+ (household size), classification_1 is Age Group1..6.
    targets = [("classification_3", True, "income band"),
               ("classification_4", True, "household size"),
               ("classification_1", True, "age group"),
               ("KID_CATEGORY_DESC", True, "children"),
               ("HOMEOWNER_DESC", True, "homeowner"),
               ("classification_5", True, "classification_5"),
               ("classification_2", True, "classification_2")]
    res["households"] = {}
    hdr = f"   {'target':22s} {'n':>5s} {'base':>6s}"
    for bn in blocks:
        hdr += f" {bn.split(' ')[0]:>10s}"
    log(hdr + f" {'shuffled':>9s}")
    for col, is_cls, pretty in targets:
        if col not in dem.columns:
            continue
        sub = dem.dropna(subset=[col])
        y = pd.factorize(sub[col])[0]
        keep = np.bincount(y) >= 5
        ok = keep[y]
        sub, y = sub[ok], y[ok]
        if len(sub) < 100 or len(np.unique(y)) < 2:
            continue
        base = float(np.bincount(y).max() / len(y))
        row = {"n": int(len(sub)), "majority_class": base}
        line = f"   {pretty:22s} {len(sub):5d} {base:6.3f}"
        for bn, X in blocks.items():
            s = cv_probe(X[sub.user_id.values], y, is_cls, seed=a.seed)
            row[bn] = s
            line += f" {s:10.3f}"
        rng = np.random.default_rng(a.seed)
        s = cv_probe(blocks["all concatenated"][sub.user_id.values],
                     rng.permutation(y), is_cls, seed=a.seed)
        row["shuffled"] = s
        res["households"][pretty] = row
        log(line + f" {s:9.3f}")
    log("   (classification accuracy, 5-fold CV; 'base' = always predict the majority)")

    # ---------------------------------------------------------------- products
    log("")
    log("2. PRODUCT EMBEDDINGS vs held-out sub-commodity labels")
    A = m.alpha.detach().numpy()
    An = A / np.maximum(np.linalg.norm(A, axis=1, keepdims=True), 1e-9)
    items = pd.read_parquet(os.path.join(IN, "items.parquet")).sort_values("item_id")
    sub = items.sub_id.to_numpy()
    rng = np.random.default_rng(a.seed)
    R = rng.normal(size=A.shape).astype(np.float32)
    Rn = R / np.linalg.norm(R, axis=1, keepdims=True)
    res["products"] = {}
    for name, E in [("fitted alpha", An), ("random, same shape", Rn)]:
        idx = rng.choice(len(E), size=min(a.n_items, len(E)), replace=False)
        S = E[idx] @ E[idx].T
        np.fill_diagonal(S, -np.inf)
        knn = np.argsort(-S, axis=1)[:, :a.k]
        purity = float((sub[idx][knn] == sub[idx][:, None]).mean())
        same = (sub[idx][:, None] == sub[idx][None, :])
        iu = np.triu_indices(len(idx), 1)
        sc, lb = S[iu], same[iu]
        order = np.argsort(-sc)
        lb = lb[order]
        tp = np.cumsum(lb); fp = np.cumsum(~lb)
        auc = float(np.trapezoid(tp / max(tp[-1], 1), fp / max(fp[-1], 1)))
        res["products"][name] = {"knn_purity": purity, "same_sub_auc": auc}
        log(f"   {name:20s} {a.k}-NN purity {purity:.4f}   same-sub AUC {auc:.4f}")
    chance = float((np.bincount(sub) / len(sub) ** 1).astype(float).dot(
        np.bincount(sub) - 1) / max(len(sub) - 1, 1))
    log(f"   chance purity (random neighbour, matched to sub-commodity sizes) {chance:.4f}")
    res["products"]["chance_purity"] = chance

    # ---------------------------------------------------------------- prices
    log("")
    log("3. PRICE SENSITIVITY  g_ij = gamma_i . beta_j")
    G = (m.gamma.detach() @ m.beta.detach().T).numpy()       # [N, J]
    log(f"   sign convention: b_ijt contains -(gamma.beta) * dlogp, so POSITIVE = "
        f"buys less when dear")
    log(f"   mean {G.mean():+.4f}  sd {G.std():.4f}  "
        f"median {np.median(G):+.4f}")
    frac_neg = float((G < 0).mean())
    log(f"   share of (household, product) pairs with the WRONG sign "
        f"(upward-sloping demand): {frac_neg:.2%}")
    hh = G.mean(1); pr = G.mean(0)
    var_h = float(hh.var()); var_p = float(pr.var()); var_t = float(G.var())
    log(f"   variance decomposition of g_ij:")
    log(f"     between households {var_h/var_t:6.1%}   between products {var_p/var_t:6.1%}"
        f"   residual {1-(var_h+var_p)/var_t:6.1%}")
    res["price"] = {"mean": float(G.mean()), "sd": float(G.std()),
                    "frac_wrong_sign": frac_neg,
                    "share_between_households": var_h / var_t,
                    "share_between_products": var_p / var_t}

    # does household price sensitivity track income, which the model never saw?
    di = dem.dropna(subset=["classification_3"]).copy()
    di["rank"] = di.classification_3.str.extract(r"(\d+)").astype(float)
    di = di.dropna(subset=["rank"])
    if len(di) > 100:
        from scipy.stats import spearmanr
        rho, p = spearmanr(di["rank"].values, hh[di.user_id.values])
        log(f"   household mean sensitivity vs income band: spearman {rho:+.4f} "
            f"(p={p:.3g}, n={len(di)})")
        log(f"     negative = poorer households are MORE price sensitive")
        res["price"]["income_spearman"] = float(rho)
        res["price"]["income_p"] = float(p)
        res["price"]["income_n"] = int(len(di))

    # choice margin vs quantity margin
    if getattr(m, "q_gamma", None) is not None:
        Q = (m.q_gamma.detach() @ m.q_beta.detach().T).numpy()
        from scipy.stats import spearmanr
        rho, _ = spearmanr(G.mean(0), Q.mean(0))
        log(f"   per-product choice margin vs quantity margin: spearman {rho:+.4f}")
        log(f"     they are fitted separately; agreement is not imposed")
        res["price"]["choice_vs_quantity_spearman"] = float(rho)
        res["price"]["quantity_mean"] = float(Q.mean())

    # most and least price-sensitive categories
    cat = d.item_cat_np
    # cat_id is a factorize() of COMMODITY_DESC in 22_basket_data, so the label is the
    # rank of the description among sorted uniques of the items table.
    items_all = pd.read_parquet(os.path.join(IN, "items.parquet"))
    cname = (items_all[["cat_id", "COMMODITY_DESC"]].drop_duplicates()
             .set_index("cat_id").COMMODITY_DESC.to_dict()
             if "COMMODITY_DESC" in items_all.columns else {})
    per_cat = pd.DataFrame({"cat": cat, "g": pr}).groupby("cat").agg(
        g=("g", "mean"), n=("g", "size"))
    per_cat = per_cat[per_cat.n >= 20].sort_values("g")
    name = cname
    log("")
    log("   least price-sensitive categories (>=20 products):")
    for c, r in per_cat.head(5).iterrows():
        log(f"     {str(name.get(c, c))[:38]:38s} {r.g:+.4f}  ({int(r.n)} products)")
    log("   most price-sensitive:")
    for c, r in per_cat.tail(5).iloc[::-1].iterrows():
        log(f"     {str(name.get(c, c))[:38]:38s} {r.g:+.4f}  ({int(r.n)} products)")
    res["price"]["cat_least"] = [[str(name.get(c, c)), float(r.g)]
                                 for c, r in per_cat.head(5).iterrows()]
    res["price"]["cat_most"] = [[str(name.get(c, c)), float(r.g)]
                                for c, r in per_cat.tail(5).iloc[::-1].iterrows()]

    with open(os.path.join(OUT, f"representation_eval_{a.label}.json"), "w") as f:
        json.dump(res, f, indent=2)
    log("")
    log(f"wrote out/representation_eval_{a.label}.json")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--label", default="spec_nested")
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--n-items", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    main(p.parse_args())
