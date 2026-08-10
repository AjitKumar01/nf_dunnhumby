"""
Test 8 of the version-3 falsifier list.

Section 5.1 removed the inventory state because dunnhumby records no inventory: purchases
are the inflow, consumption is never observed, and the opening stock is never observed.  In
its place Eq. 4 defines a days-of-supply proxy

    x_sup = Q_ic  -  dt_ic / g_c

built only from observables: Q_ic the units of category c taken on the household's last
occasion in c, dt_ic the days since, and g_c the category's median repurchase gap.

Version 2's recency features already carry dt and g.  The ONLY new information in Eq. 4 is
Q -- how much was bought, not just when.  So the whole question is:

    does the SIZE of the last purchase predict the next one, given its TIMING?

If not, Eq. 4 buys nothing, it should be dropped, and forward-buying is not representable in
this dataset -- which matters, because a markdown policy evaluated in an environment that
cannot represent pull-forward will see the sales lift and never the payback.

TWO DESIGNS, because they fail differently.

  A -- the mechanism.  Regress the gap to the household's NEXT purchase of category c on
       the units taken now, within (household, category) so the baseline rate is absorbed,
       and controlling for the trip's total size so "it was a big shop" is not the driver.
       Forward-buying predicts a POSITIVE coefficient: more now, longer until next.

  B -- incremental prediction, held out.  Does adding Q to a flexible function of dt improve
       out-of-sample prediction of whether the category is bought on this trip?  Fitted on
       training weeks, scored on test weeks, in held-out log-loss.

Design A can be significant while B is negligible: a real but tiny mechanism.  B is what
decides whether the feature earns a place in the model.

Writes out/test8_forward_buying.json.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")


def log(m):
    print(f"[t8] {m}", flush=True)


def occasions():
    """One row per (household, category, trip) on which the category was bought."""
    b = pd.read_parquet(os.path.join(ROOT, "basket_input", "baskets.parquet"))
    it = pd.read_parquet(os.path.join(ROOT, "basket_input", "items.parquet"))[
        ["item_id", "cat_id"]]
    b = b.merge(it, on="item_id", how="left")
    trip_size = b.groupby("BASKET_ID").size().rename("n_lines")
    occ = (b.groupby(["user_id", "cat_id", "BASKET_ID", "DAY", "WEEK_NO", "split"])
             .agg(Q=("units", "sum"), k=("item_id", "size")).reset_index()
             .merge(trip_size, on="BASKET_ID"))
    occ = occ.sort_values(["user_id", "cat_id", "DAY"])
    g = occ.groupby(["user_id", "cat_id"])
    occ["gap_next"] = g["DAY"].shift(-1) - occ["DAY"]
    occ["dt_prev"] = occ["DAY"] - g["DAY"].shift(1)
    occ["Q_prev"] = g["Q"].shift(1)
    return occ


def within(df, keys, cols):
    """Subtract the (household, category) mean from each column -- the within transform."""
    out = df.copy()
    grp = out.groupby(keys)
    for c in cols:
        out[c + "_w"] = out[c] - grp[c].transform("mean")
    return out


def ols(X, y):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    n, k = X.shape
    xtx_inv = np.linalg.pinv(X.T @ X)
    s2 = float(resid @ resid) / max(n - k, 1)
    return beta, np.sqrt(np.diag(xtx_inv) * s2), resid


def cluster_se(X, resid, groups):
    """Household-clustered standard errors -- trips repeat within households."""
    xtx_inv = np.linalg.pinv(X.T @ X)
    meat = np.zeros((X.shape[1], X.shape[1]))
    order = np.argsort(groups)
    gs, Xs, rs = groups[order], X[order], resid[order]
    bounds = np.flatnonzero(np.diff(gs)) + 1
    for sl in np.split(np.arange(len(gs)), bounds):
        u = Xs[sl].T @ rs[sl]
        meat += np.outer(u, u)
    V = xtx_inv @ meat @ xtx_inv
    return np.sqrt(np.diag(V))


def design_a(occ, res):
    log("=== DESIGN A: does buying more now push the next purchase further out? ===")
    d = occ[(occ.split == "train") & occ.gap_next.notna()].copy()
    d = d[d.gap_next > 0]
    cnt = d.groupby(["user_id", "cat_id"])["Q"].transform("size")
    d = d[cnt >= 3]                       # need within-cell variation
    log(f"  {len(d):,} occasions over "
        f"{d.groupby(['user_id','cat_id']).ngroups:,} (household, category) cells")
    log(f"  units per occasion: mean {d.Q.mean():.3f}  sd {d.Q.std():.3f}  "
        f"share Q>1 {float((d.Q > 1).mean()):.3f}")
    d["y"] = np.log(d.gap_next.values)
    d = within(d, ["user_id", "cat_id"], ["y", "Q", "n_lines"])
    out = {}
    for name, cols in [("Q alone", ["Q_w"]), ("Q + trip size", ["Q_w", "n_lines_w"])]:
        X = d[cols].to_numpy(float)
        beta, _, resid = ols(X, d["y_w"].to_numpy(float))
        se = cluster_se(X, resid, d["user_id"].to_numpy())
        log(f"  within (household, category), outcome log(gap to next purchase)")
        log(f"    {name:16s} beta_Q {beta[0]:+.5f}  se {se[0]:.5f}  "
            f"t {beta[0]/se[0]:+.2f}")
        out[name] = dict(beta=float(beta[0]), se=float(se[0]),
                         t=float(beta[0] / se[0]))
    b = out["Q + trip size"]["beta"]
    log(f"  one extra unit lengthens the next gap by {100*b:.2f}% "
        f"(mean gap {d.gap_next.mean():.1f} days -> +{d.gap_next.mean()*b:.2f} days)")
    res["design_a"] = out
    res["design_a"]["mean_gap_days"] = float(d.gap_next.mean())
    res["design_a"]["n"] = int(len(d))


def design_b(occ, res, a):
    log("")
    log("=== DESIGN B: held-out incremental prediction of category incidence ===")
    b = pd.read_parquet(os.path.join(ROOT, "basket_input", "baskets.parquet"),
                        columns=["BASKET_ID", "user_id", "DAY", "WEEK_NO", "split"])
    trips = b.drop_duplicates("BASKET_ID").sort_values(["user_id", "DAY"])
    bought = occ[["user_id", "cat_id", "DAY", "Q"]].copy()
    rng = np.random.default_rng(a.seed)

    # per (household, category): the history of purchase days and units
    hist = {k: (v.DAY.to_numpy(), v.Q.to_numpy())
            for k, v in bought.groupby(["user_id", "cat_id"])}
    gmed = (bought.sort_values(["user_id", "cat_id", "DAY"])
            .assign(gap=lambda x: x.groupby(["user_id", "cat_id"]).DAY.diff())
            .groupby("cat_id")["gap"].median())

    rows = []
    keep = trips.sample(min(a.trips, len(trips)), random_state=a.seed)
    by_hh = {}
    for (u, c), (days, qs) in hist.items():
        by_hh.setdefault(u, []).append((c, days, qs))
    for r in keep.itertuples():
        for c, days, qs in by_hh.get(r.user_id, []):
            i = np.searchsorted(days, r.DAY)          # strictly-before lookup
            if i == 0:
                continue
            rows.append((r.user_id, c, r.DAY - days[i - 1], qs[i - 1],
                         1 if (i < len(days) and days[i] == r.DAY) else 0,
                         float(gmed.get(c, 26.0)), r.split))
    P = pd.DataFrame(rows, columns=["user_id", "cat_id", "dt", "Q_prev", "y", "g", "split"])
    P = P[P.dt > 0]
    log(f"  {len(P):,} (trip, previously-bought category) rows from "
        f"{len(keep):,} sampled trips; base rate {P.y.mean():.4f}")

    tr, te = P[P.split == "train"], P[P.split == "test"]
    log(f"  train {len(tr):,}  test {len(te):,}")

    def feats(D, with_Q):
        z = np.clip(D.dt.to_numpy(float) / D.g.to_numpy(float), 0, 8)
        cols = [np.ones(len(D)), np.log1p(D.dt.to_numpy(float)),
                np.exp(-D.dt.to_numpy(float) / 7.0), np.exp(-z), z, z ** 2]
        if with_Q:
            q = D.Q_prev.to_numpy(float)
            cols += [q, q / D.g.to_numpy(float),
                     q - D.dt.to_numpy(float) / D.g.to_numpy(float)]   # Eq. 4 itself
        return np.column_stack(cols)

    def logit_fit(X, y, iters=60, lam=1e-4):
        w = np.zeros(X.shape[1])
        for _ in range(iters):
            p = 1.0 / (1.0 + np.exp(-X @ w))
            W = np.clip(p * (1 - p), 1e-9, None)
            g = X.T @ (y - p) - lam * w
            H = (X * W[:, None]).T @ X + lam * np.eye(X.shape[1])
            w += np.linalg.solve(H, g)
        return w

    out = {}
    for name, wq in [("dt only", False), ("dt + Q", True)]:
        Xtr, Xte = feats(tr, wq), feats(te, wq)
        w = logit_fit(Xtr, tr.y.to_numpy(float))
        p = 1.0 / (1.0 + np.exp(-Xte @ w))
        p = np.clip(p, 1e-12, 1 - 1e-12)
        yv = te.y.to_numpy(float)
        ll = float(-(yv * np.log(p) + (1 - yv) * np.log(1 - p)).mean())
        out[name] = ll
        log(f"    {name:10s} held-out log-loss {ll:.6f}")
    gain = out["dt only"] - out["dt + Q"]
    base = float(te.y.mean())
    null = float(-(base * np.log(base) + (1 - base) * np.log(1 - base)))
    log(f"  improvement from Q: {gain:.6f} nats "
        f"({100*gain/(null - out['dt only']):.2f}% of what dt itself buys over the base rate)")
    res["design_b"] = dict(ll_dt=out["dt only"], ll_dt_Q=out["dt + Q"], gain=gain,
                           null_ll=null, n_test=int(len(te)))


def main(a):
    occ = occasions()
    res = {}
    design_a(occ, res)
    design_b(occ, res, a)
    log("")
    ga = res["design_a"]["Q + trip size"]
    gb = res["design_b"]["gain"]
    log("VERDICT")
    log(f"  mechanism: beta_Q {ga['beta']:+.5f} (t {ga['t']:+.1f}) -- "
        f"{'present' if abs(ga['t']) > 3 else 'not detected'}"
        f"{', and in the forward-buying direction' if ga['beta'] > 0 else ''}")
    log(f"  prediction: {gb:+.6f} nats held out -- "
        f"{'worth keeping' if gb > 1e-4 else 'negligible'}")
    with open(os.path.join(ROOT, "out", "test8_forward_buying.json"), "w") as fh:
        json.dump(res, fh, indent=2)
    log("wrote out/test8_forward_buying.json")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--trips", type=int, default=12000)
    p.add_argument("--seed", type=int, default=0)
    main(p.parse_args())
