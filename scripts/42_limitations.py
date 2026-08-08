"""
Stage 42 -- Attack three stated limitations instead of restating them.

A. ASSORTMENT IS SELECTION ON THE OUTCOME.  C_s(c) is "sold at least once in training",
   so the choice set is built from the choices.  Refitting under a different rule is
   expensive; the cheap and nearly as informative test is to REBUILD THE CHOICE SET AT
   EVALUATION TIME under stricter thresholds, holding the fitted model fixed, and see
   whether the estimand moves.  If the price coefficient and the implied elasticity are
   flat across thresholds, the selection is not driving them.

B. GENERATED BASKETS OVER-PRODUCE NOVELTY.  46.1% of generated purchases are products the
   household never bought, against 14.1% real, and repeat purchases are staler (median 90
   days against 32).  That is a symptom of the item conditional being too FLAT: sampling
   from a diffuse softmax produces variety a real shopper does not show.  If so, a single
   temperature on the generation softmax should fix the novelty rate -- and the test is
   whether it fixes it WITHOUT breaking the statistics that currently match.

C. NO JOINT VALIDATION.  Every likelihood is a conditional and Section 6 compares
   summary statistics one at a time, which cannot detect a joint mismatch.  A classifier
   two-sample test does: train a discriminator to tell real baskets from generated ones on
   basket-level features.  AUC 0.5 means indistinguishable; anything above says exactly
   which features give the generator away.

Writes out/limitations_<label>.json.
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
DATA = os.path.join(HERE, "..", "data")
OUT = os.path.join(HERE, "..", "out")


def log(m):
    print(f"[42] {m}", flush=True)


def main(a):
    dev = torch.device("cpu")
    d = nb.NestedData(IN, device=dev)
    m, _ = cf.load(a.label, d, dev)
    sp = d.splits["test"]
    rng = np.random.default_rng(a.seed)
    res = {"label": a.label}

    # ================================================================= A
    log("")
    log("A. ASSORTMENT THRESHOLD -- does the estimand move when the choice set changes?")
    items = pd.read_parquet(os.path.join(IN, "items.parquet"))
    tx = pd.read_parquet(os.path.join(DATA, "tx.parquet"),
                         columns=["PRODUCT_ID", "STORE_ID", "WEEK_NO"])
    tx = tx.merge(items[["PRODUCT_ID", "item_id"]], on="PRODUCT_ID")
    meta = json.load(open(os.path.join(IN, "meta.json")))
    val_from = meta.get("val_from", 83)
    tx = tx[tx.WEEK_NO < val_from]
    stores = np.sort(tx.STORE_ID.unique())
    sidx = pd.Series(np.arange(len(stores)), index=stores)
    tx["sid"] = tx.STORE_ID.map(sidx)
    tx = tx[tx.sid < d.carried.shape[1]]
    cnt = tx.groupby(["item_id", "sid"]).size()
    ij = np.array(cnt.index.tolist())
    sales = np.zeros(tuple(d.carried.shape), dtype=np.int32)
    sales[ij[:, 0], ij[:, 1]] = cnt.to_numpy()
    log(f"   sales matrix built: {(sales > 0).mean():.1%} of the product x store grid "
        f"has >=1 sale")

    bidx = rng.choice(sp["n_baskets"], size=min(a.n_baskets, sp["n_baskets"]),
                      replace=False)
    rows = np.concatenate([np.arange(sp["starts"][i], sp["ends"][i]) for i in bidx])
    user, item = sp["user"][rows], sp["item"][rows]
    day, week, store = sp["day"][rows], sp["week"][rows], sp["store"][rows]
    B = len(rows)
    cats_r = d.item_cat_np[item]
    cand = d.cat_items_np[cats_r]
    M = cand.shape[1]
    tgt = torch.as_tensor(d.item_pos_np[item])
    day_r = np.repeat(day[:, None], M, 1)
    user_r = np.repeat(user[:, None], M, 1)
    store_r = np.repeat(store[:, None], M, 1)
    rw = np.repeat(sp["raw_week"][rows][:, None], M, 1)
    st = torch.as_tensor(d.state(user_r.ravel(), cand.ravel(),
                                 day_r.ravel()).reshape(B, M, nb.N_STATE_FEATURES))
    ci = torch.as_tensor(cand)
    dl = d.log_price_dev[ci, torch.as_tensor(day_r)]
    if m.use_store and m.use_store_price:
        dl = dl + d.store_dev(cand.ravel(), store_r.ravel(), rw.ravel()).reshape(B, M)
    A_ = m.alpha.detach()
    ctx = torch.zeros(B, m.K)
    own = np.concatenate([[k] * (sp["ends"][i] - sp["starts"][i])
                          for k, i in enumerate(bidx)])
    for b in np.unique(own):
        r = np.flatnonzero(own == b)
        if len(r) > 1:
            tot = A_[torch.as_tensor(item[r])].sum(0)
            for q, rr in enumerate(r):
                ctx[rr] = (tot - A_[item[r[q]]]) / (len(r) - 1)
    with torch.no_grad():
        u = m.item_utility(torch.as_tensor(user), ci, ctx, dl, st,
                           torch.as_tensor(week), torch.as_tensor(store))
    base_mask = torch.as_tensor(d.cat_mask_np[cats_r]) > 0
    ar = torch.arange(B)
    log("")
    log(f"   {'min sales to be in the choice set':34s} {'set size':>9s} "
        f"{'item loglik':>12s} {'top-1':>7s} {'elasticity':>11s}")
    tab = []
    for thr in a.thresholds:
        keep = torch.as_tensor(sales[cand, store_r] >= thr) & base_mask
        keep[ar, tgt] = True
        uu = u.masked_fill(~keep, -1e9)
        lp = torch.log_softmax(uu, 1)
        ll = float(lp[ar, tgt].mean())
        t1 = float((uu.argmax(1) == tgt).float().mean())
        pi = torch.softmax(uu, 1)[ar, tgt]
        gb = (m.gamma.detach()[torch.as_tensor(user)]
              * m.beta.detach()[torch.as_tensor(item)]).sum(-1)
        el = float((-gb * (1 - pi)).median())
        n = float(keep.float().sum(1).mean())
        tab.append({"threshold": thr, "set_size": n, "loglik": ll,
                    "top1": t1, "allocation_elasticity": el})
        log(f"   {'>= ' + str(thr) + ' sales':34s} {n:9.1f} {ll:12.4f} {t1:7.3f} "
            f"{el:11.4f}")
    res["assortment"] = tab
    log("   (the fitted model is held fixed; only the choice set changes)")

    # ================================================================= B
    log("")
    log("B. NOVELTY -- is the item conditional simply too flat?")
    log(f"   generating at several softmax temperatures; T<1 sharpens")
    real_rows = np.concatenate([np.arange(sp["starts"][i], sp["ends"][i])
                                for i in range(min(a.n_trips, sp["n_baskets"]))])
    sr = d.state(sp["user"][real_rows], sp["item"][real_rows], sp["day"][real_rows])
    real_novel = float(sr[:, 0].mean())
    real_size = float(np.mean([sp["ends"][i] - sp["starts"][i]
                               for i in range(min(a.n_trips, sp["n_baskets"]))]))
    log("")
    log(f"   {'temperature':>12s} {'novel %':>9s} {'items/basket':>13s} "
        f"{'median days since':>18s}")
    log(f"   {'REAL':>12s} {real_novel:9.2%} {real_size:13.2f} "
        f"{np.median(np.expm1(sr[sr[:,0]==0,3]*np.log(100))):18.0f}")
    tabB = []
    for T in a.temps:
        orig = m.item_utility
        if T != 1.0:
            def scaled(*args, _f=orig, _T=T, **kw):
                return _f(*args, **kw) / _T
            m.item_utility = scaled
        g = cf.generate_baskets(m, d, dev, n_trips=a.n_trips, seed=a.seed,
                                sweeps=4, use_ctx=True, with_units=False)
        m.item_utility = orig
        gi = np.concatenate([np.asarray(b) for b in g if len(b)])
        gu = np.concatenate([np.full(len(b), int(sp["user"][sp["starts"][k]]))
                             for k, b in enumerate(g) if len(b)])
        gd = np.concatenate([np.full(len(b), int(sp["day"][sp["starts"][k]]))
                             for k, b in enumerate(g) if len(b)])
        sg = d.state(gu, gi, gd)
        nov = float(sg[:, 0].mean())
        sz = float(np.mean([len(b) for b in g]))
        med = float(np.median(np.expm1(sg[sg[:, 0] == 0, 3] * np.log(100))))
        tabB.append({"T": T, "novel": nov, "items": sz, "median_days": med})
        log(f"   {T:12.2f} {nov:9.2%} {sz:13.2f} {med:18.0f}")
    res["novelty"] = {"real_novel": real_novel, "real_items": real_size,
                      "by_temperature": tabB}

    # ================================================================= C
    log("")
    log("C. CLASSIFIER TWO-SAMPLE TEST -- can a discriminator tell them apart?")
    g = cf.generate_baskets(m, d, dev, n_trips=a.n_trips, seed=a.seed + 1,
                            sweeps=4, use_ctx=True, with_units=False)
    cnt_tr = np.bincount(d.splits["train"]["item"], minlength=d.J).astype(float)
    pop = np.log1p(cnt_tr)
    An = m.alpha.detach().numpy()
    An = An / np.maximum(np.linalg.norm(An, axis=1, keepdims=True), 1e-9)

    def feats(baskets):
        F = []
        for b in baskets:
            b = np.asarray(b)
            if len(b) == 0:
                continue
            c = d.item_cat_np[b]
            sim = 0.0
            if len(b) > 1:
                S = An[b] @ An[b].T
                iu = np.triu_indices(len(b), 1)
                sim = float(S[iu].mean())
            F.append([len(b), len(np.unique(c)), len(b) / max(len(np.unique(c)), 1),
                      pop[b].mean(), pop[b].std(), pop[b].min(), sim,
                      float(np.bincount(c).max())])
        return np.array(F)

    Xr = feats([sp["item"][sp["starts"][i]:sp["ends"][i]]
                for i in range(min(a.n_trips, sp["n_baskets"]))])
    Xg = feats(g)
    n = min(len(Xr), len(Xg))
    X = np.vstack([Xr[:n], Xg[:n]])
    y = np.r_[np.zeros(n), np.ones(n)]
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import cross_val_predict, StratifiedKFold
    from sklearn.metrics import roc_auc_score
    clf = GradientBoostingClassifier(n_estimators=120, max_depth=3, random_state=a.seed)
    pr = cross_val_predict(clf, X, y, cv=StratifiedKFold(5, shuffle=True,
                                                         random_state=a.seed),
                           method="predict_proba")[:, 1]
    auc = float(roc_auc_score(y, pr))
    log(f"   {n:,} real vs {n:,} generated baskets, 8 basket-level features")
    log(f"   discriminator AUC = {auc:.4f}   (0.5 = indistinguishable)")
    clf.fit(X, y)
    names = ["n items", "n categories", "items per category", "mean log-popularity",
             "sd log-popularity", "min log-popularity", "mean pairwise cos(alpha)",
             "largest category count"]
    imp = sorted(zip(names, clf.feature_importances_), key=lambda x: -x[1])
    log("   what gives it away:")
    for nme, v in imp[:5]:
        rm, gm = Xr[:, names.index(nme)].mean(), Xg[:, names.index(nme)].mean()
        log(f"     {nme:26s} importance {v:.3f}   real {rm:8.3f}  generated {gm:8.3f}")
    res["c2st"] = {"auc": auc, "n": int(n),
                   "importances": [[k, float(v)] for k, v in imp],
                   "real_means": {k: float(Xr[:, names.index(k)].mean()) for k in names},
                   "gen_means": {k: float(Xg[:, names.index(k)].mean()) for k in names}}

    with open(os.path.join(OUT, f"limitations_{a.label}.json"), "w") as f:
        json.dump(res, f, indent=2)
    log("")
    log(f"wrote out/limitations_{a.label}.json")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--label", default="spec_nested")
    p.add_argument("--thresholds", type=int, nargs="+", default=[1, 2, 3, 5, 10, 25])
    p.add_argument("--temps", type=float, nargs="+", default=[1.0, 0.7, 0.5, 0.35, 0.25])
    p.add_argument("--n-baskets", type=int, default=2500)
    p.add_argument("--n-trips", type=int, default=5000)
    p.add_argument("--seed", type=int, default=0)
    main(p.parse_args())
