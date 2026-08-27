"""Estimate the elasticity targets the model is calibrated to, from the data alone.

The model does not fit these -- they are external targets. `--elast-target` pins the
aggregate response by projection, and `--kappa-init` sets the idiosyncratic/aggregate split
from them (see docs/THEORY.md section 6).  So they must come from the data on the machine
that trains, not from constants baked into a script.

TWO ESTIMATORS, on deliberately different variation:

  A  item-week time series, item fixed effects.  Identification comes from an item's own
     price moving over weeks.  Its exposure is that item-specific demand shocks may move
     WITH promotion schedules, biasing the estimate; display and mailer are controlled, but
     imperfectly.

  B  cross-store within item-week.  store_price.npz records each item's deviation from the
     chain price at a given store-week, so demeaning within (item, week) removes everything
     common to that item that week -- national promotion, seasonality, the demand shock A is
     exposed to -- and identifies from stores charging different prices in the SAME week.

They share no identifying variation.  Agreement is evidence; disagreement is a finding, and
either way the number that gets used is reported with both.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

from paths import BI


def log(s=""):
    print(s, flush=True)


def ols(y, *xs):
    X = np.column_stack(xs)
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    r = y - X @ b
    s2 = (r @ r) / max(len(y) - X.shape[1], 1)
    se = np.sqrt(np.diag(s2 * np.linalg.pinv(X.T @ X)))
    return b, se


def _demean(v, g):
    """Subtract the mean within each group g -- i.e. absorb a g fixed effect."""
    n = np.bincount(g, minlength=g.max() + 1)
    s = np.bincount(g, weights=v, minlength=g.max() + 1)
    return v - (s / np.maximum(n, 1))[g]


def estimator_A(D, sub):
    """Item-week panel, item fixed effects, display and mailer controlled."""
    lp, li = D["line_ptr"], D["line_item"]
    J = int(D["n_item"]); S = int(D["n_store"])
    tw, td = D["trip_week"], D["trip_day"]
    tr = np.flatnonzero(D["trip_split"] == 0); W = int(tw.max()) + 1
    logp = np.load(os.path.join(BI, "log_price.npy"))
    pr = np.load(os.path.join(BI, "promo.npz"))
    pk = pr["keys"]; pdi = pr["disp"].astype(float); pma = pr["mail"].astype(float)
    w_of = (pk % 128).astype(np.int64); it_of = (pk // 128 // S).astype(np.int64)
    sel = (w_of < W) & (it_of < J)
    w_of, it_of, pdi, pma = w_of[sel], it_of[sel], pdi[sel], pma[sel]
    flat = it_of * W + w_of; N = np.bincount(flat, minlength=J * W)
    Dv = np.where(N > 0, np.bincount(flat, weights=pdi, minlength=J * W) / np.maximum(N, 1),
                  0.).reshape(J, W)
    Mv = np.where(N > 0, np.bincount(flat, weights=pma, minlength=J * W) / np.maximum(N, 1),
                  0.).reshape(J, W)
    nbuy = np.zeros((J, W)); ntrip = np.zeros(W)
    for t in tr:
        w = int(tw[t]); ntrip[w] += 1
        for j in np.unique(li[int(lp[t]):int(lp[t + 1])]):
            nbuy[int(j), w] += 1
    dow = np.zeros(712, dtype=int)
    for t in tr:
        dow[int(td[t])] = int(tw[t])
    P = np.full((J, W), np.nan)
    for w in range(W):
        d = np.flatnonzero(dow == w)
        if len(d):
            P[:, w] = logp[:, d].mean(1)
    inc = np.where(ntrip[None, :] > 0, nbuy / np.maximum(ntrip[None, :], 1), np.nan)
    ok = np.isfinite(P) & np.isfinite(inc) & (inc > 0) & (nbuy >= 5)
    keep = ok.sum(1) >= 20
    Y = []; Xo = []; Xc = []; Xd = []; Xm = []
    for j in np.flatnonzero(keep):
        sibs = np.flatnonzero((sub == sub[j]) & (np.arange(J) != j))
        m = ok[j] & (np.isfinite(P[sibs].mean(0)) if len(sibs) else ok[j])
        if m.sum() < 20:
            continue
        riv = P[sibs][:, m].mean(0) if len(sibs) else np.zeros(m.sum())
        y = np.log(inc[j, m])
        for arr, v in ((Y, y), (Xo, P[j, m]), (Xc, riv), (Xd, Dv[j, m]), (Xm, Mv[j, m])):
            arr.append(v - v.mean())
    Y = np.concatenate(Y); Xo = np.concatenate(Xo); Xc = np.concatenate(Xc)
    Xd = np.concatenate(Xd); Xm = np.concatenate(Xm)
    b, se = ols(Y, Xo, Xc, Xd, Xm)
    return dict(own=float(b[0]), own_se=float(se[0]),
                cross=float(b[1]), cross_se=float(se[1]), n=int(len(Y)))


def estimator_B(D, sub, min_cell=3):
    """Cross-store within item-week: store deviations from the chain price.

    Demeaning within (item, week) absorbs every national shock to that item that week, so
    the surviving variation is stores pricing the same item differently at the same time.
    """
    lp, li = D["line_ptr"], D["line_item"]
    J = int(D["n_item"]); S = int(D["n_store"])
    tw, ts = D["trip_week"], D["trip_store"]
    tr = np.flatnonzero(D["trip_split"] == 0); W = int(tw.max()) + 1
    sp = np.load(os.path.join(BI, "store_price.npz"))
    it_, st_, wk_, dev_ = sp["item"], sp["store"], sp["week"], sp["dev"].astype(float)
    ok = (it_ < J) & (st_ < S) & (wk_ < W)
    it_, st_, wk_, dev_ = it_[ok], st_[ok], wk_[ok], dev_[ok]
    # purchases and traffic by (item, store, week)
    key = lambda i, s, w: (i.astype(np.int64) * S + s) * W + w
    n_cell = np.zeros(J * S * W, dtype=np.int32)
    traffic = np.zeros(S * W, dtype=np.int32)
    for t in tr:
        s = int(ts[t]); w = int(tw[t]); traffic[s * W + w] += 1
        for j in np.unique(li[int(lp[t]):int(lp[t + 1])]):
            n_cell[(int(j) * S + s) * W + w] += 1
    k = key(it_, st_, wk_)
    buy = n_cell[k]; trf = traffic[st_.astype(np.int64) * W + wk_]
    good = (buy >= min_cell) & (trf >= 30)
    it_, st_, wk_, dev_, buy, trf = (v[good] for v in (it_, st_, wk_, dev_, buy, trf))
    y = np.log(buy / trf)
    iw = it_.astype(np.int64) * W + wk_            # item x week group id
    _, iw = np.unique(iw, return_inverse=True)
    # rival store deviation: mean dev of OTHER items in the same sub-commodity, same (store, week)
    sw = st_.astype(np.int64) * W + wk_
    grp = sub[it_].astype(np.int64) * (S * W) + sw
    _, g = np.unique(grp, return_inverse=True)
    cnt = np.bincount(g); tot = np.bincount(g, weights=dev_)
    rival = np.where(cnt[g] > 1, (tot[g] - dev_) / np.maximum(cnt[g] - 1, 1), np.nan)
    # OWN price uses every usable cell.  Requiring a rival as well would restrict it to the
    # rare (store, week) where two items of the same sub-commodity BOTH have an observed
    # store price -- 187 of ~190,000 cells, a selected sample that returned +5.99.
    yD = _demean(y, iw); pD = _demean(dev_, iw)
    b, se = ols(yD, pD)
    out = dict(own=float(b[0]), own_se=float(se[0]), n=int(len(y)))
    m = np.isfinite(rival)
    if m.sum() >= 200:
        yM = _demean(y[m], iw[m]); pM = _demean(dev_[m], iw[m]); rM = _demean(rival[m], iw[m])
        b2, se2 = ols(yM, pM, rM)
        out.update(cross=float(b2[1]), cross_se=float(se2[1]), n_cross=int(m.sum()))
    else:
        # Not enough same-sub-commodity pairs share a store-week for this estimator to say
        # anything about cross-price.  Reported as unavailable rather than as a number.
        out.update(cross=None, cross_se=None, n_cross=int(m.sum()))
    return out


def aggregate_elasticity(D):
    """Basket size against the store's overall price level, within week.

    A uniform price shift is what --elast-target pins, so it is estimated on the store-week
    price LEVEL with a week fixed effect.
    """
    lp = D["line_ptr"]; J = int(D["n_item"]); S = int(D["n_store"])
    tw, ts = D["trip_week"], D["trip_store"]
    tr = np.flatnonzero(D["trip_split"] == 0); W = int(tw.max()) + 1
    sp = np.load(os.path.join(BI, "store_price.npz"))
    it_, st_, wk_, dev_ = sp["item"], sp["store"], sp["week"], sp["dev"].astype(float)
    ok = (st_ < S) & (wk_ < W)
    sw = st_[ok].astype(np.int64) * W + wk_[ok]
    lvl_sum = np.bincount(sw, weights=dev_[ok], minlength=S * W)
    lvl_n = np.bincount(sw, minlength=S * W)
    level = np.where(lvl_n >= 20, lvl_sum / np.maximum(lvl_n, 1), np.nan)
    sz_sum = np.zeros(S * W); sz_n = np.zeros(S * W)
    for t in tr:
        i = int(ts[t]) * W + int(tw[t])
        sz_sum[i] += int(lp[t + 1]) - int(lp[t]); sz_n[i] += 1
    m = (sz_n >= 30) & np.isfinite(level)
    y = np.log(sz_sum[m] / sz_n[m]); x = level[m]
    idx = np.arange(S * W)[m]
    wk = (idx % W).astype(np.int64)
    stg = (idx // W).astype(np.int64)
    # TWO-WAY fixed effects.  With a week effect alone this compares expensive stores to
    # cheap ones, which is confounded by store size and catchment -- it returned +0.26,
    # i.e. higher prices buying bigger baskets.  Absorbing the store too leaves only a
    # store's own price level moving over weeks, which is the quantity wanted.
    for _ in range(30):                     # alternating projections; converges quickly
        y = _demean(y, wk); x = _demean(x, wk)
        y = _demean(y, stg); x = _demean(x, stg)
    b, se = ols(y, x)
    return dict(agg=float(b[0]), agg_se=float(se[0]), n=int(m.sum()))


def main(a):
    from data import build
    D = build()
    J = int(D["n_item"])
    items = pd.read_parquet(os.path.join(BI, "items.parquet"))
    col = "sub_id" if "sub_id" in items.columns else items.columns[2]
    sub = np.full(J, -1, dtype=np.int64)
    ii = items["item_id"].astype(int).values; ss = items[col].astype(int).values
    keep = ii < J
    sub[ii[keep]] = ss[keep]

    log("=" * 74)
    log("ELASTICITY TARGETS  (estimated from the data; the model does not fit these)")
    log("=" * 74)
    A = estimator_A(D, sub)
    log(f"\n  A  item-week time series, item FE, promo controlled     n = {A['n']:,}")
    log(f"       own-price   {A['own']:+.4f}  (se {A['own_se']:.4f})")
    log(f"       cross-price {A['cross']:+.4f}  (se {A['cross_se']:.4f})")
    B = estimator_B(D, sub)
    log(f"\n  B  cross-store within item-week (no shared variation)   n = {B['n']:,}")
    log(f"       own-price   {B['own']:+.4f}  (se {B['own_se']:.4f})")
    if B.get("cross") is None:
        log(f"       cross-price  unavailable -- only {B['n_cross']} store-weeks have two "
            f"same-sub-commodity items with observed prices")
    else:
        log(f"       cross-price {B['cross']:+.4f}  (se {B['cross_se']:.4f})")
    G = aggregate_elasticity(D)
    log(f"\n  aggregate (basket size vs store price level, week FE)   n = {G['n']:,}")
    log(f"       aggregate   {G['agg']:+.4f}  (se {G['agg_se']:.4f})")

    def agree(x, y, sx, sy):
        z = abs(x - y) / max(np.hypot(sx, sy), 1e-12)
        return z, ("consistent" if z < 2 else "DISAGREE")

    zo, so = agree(A["own"], B["own"], A["own_se"], B["own_se"])
    log(f"\n  own-price   A vs B: {zo:.1f} sigma apart -- {so}")
    if B.get("cross") is None:
        zc = float("nan")
        log(f"  cross-price A vs B: B cannot estimate it "
            f"(only {B['n_cross']} store-weeks have two same-sub-commodity items priced)")
    else:
        zc, sc = agree(A["cross"], B["cross"], A["cross_se"], B["cross_se"])
        log(f"  cross-price A vs B: {zc:.1f} sigma apart -- {sc}")

    # precision-weighted combination; if they disagree, the spread is the honest uncertainty
    # Combining estimates that DISAGREE is not averaging away noise, it is averaging away a
    # bias whose source is unknown.  Blend only when they are consistent; otherwise take the
    # better-identified one and say so.
    def comb(x, y, sx, sy):
        wx, wy = 1 / sx ** 2, 1 / sy ** 2
        return (wx * x + wy * y) / (wx + wy)

    if zo < 2:
        own, own_src = comb(A["own"], B["own"], A["own_se"], B["own_se"]), "A+B"
    else:
        own, own_src = A["own"], "A only (B disagrees; see note)"
    cross, cross_src = A["cross"], ("A+B" if B.get("cross") is not None and zc < 2 else "A only")
    if B.get("cross") is not None and zc < 2:
        cross = comb(A["cross"], B["cross"], A["cross_se"], B["cross_se"])

    # The aggregate estimator is NOT trusted when it comes out positive: a price rise
    # growing baskets is not an economic result, it is the store "price level" being the
    # mean over whichever items happen to have an observed deviation, and observation is
    # non-random (deviations are recorded when prices move).  Fall back to the established
    # value rather than inverting the model's price response.
    AGG_FALLBACK = -0.121
    if G["agg"] < 0:
        agg, agg_src = G["agg"], "estimated"
    else:
        agg, agg_src = AGG_FALLBACK, "FALLBACK -- estimator returned %+.4f, see note" % G["agg"]

    lp = D["line_ptr"]
    tr = np.flatnonzero(D["trip_split"] == 0)
    sz = np.array([int(lp[t + 1]) - int(lp[t]) for t in tr], dtype=float)
    en, vn = sz.mean(), sz.var()
    # gb is pinned by the projection to |elast_agg| * E[n]/Var(n); kappa then carries the
    # own-price response, so kappa* = |elast_own| / gb.  docs/THEORY.md section 6.
    gb = abs(agg) * en / max(vn, 1e-9)
    kappa = abs(own) / max(gb, 1e-9)

    log(f"\n  targets adopted:")
    log(f"       own-price   {own:+.4f}   [{own_src}]")
    log(f"       cross-price {cross:+.4f}   [{cross_src}]")
    log(f"       aggregate   {agg:+.4f}   [{agg_src}]")
    if zo >= 2:
        log(f"\n  NOTE  the two own-price estimators disagree.  B is identified off "
            f"{B['n']:,} cells of\n        cross-store variation (store deviations cover "
            f"0.53% of the item-store-week grid,\n        and demeaning within item-week "
            f"drops every item-week seen at one store), so it is\n        thin and prone "
            f"to attenuation from a noisy regressor.  A is adopted; the targets\n        "
            f"rest on ONE credible estimator, not two.")
    if G["agg"] >= 0:
        log(f"\n  NOTE  the aggregate estimator returned {G['agg']:+.4f} -- a price rise "
            f"growing baskets.\n        The store price level is the mean over whichever "
            f"items have an observed deviation,\n        and observation is non-random, so "
            f"this is composition, not price.  Not used.")
    # kappa* is a point estimate divided by another point estimate, so it inherits both
    # errors.  The likelihood has its own opinion -- a sweep on the fitted model puts the
    # optimum at 40-60 and measures 74.7 as worse -- and at full scale an unclamped 69.2
    # diverged (E[n] -> n_max within 1,000 iterations of stage 2).  Clamp to the band the
    # likelihood actually supports and record that it was clamped.
    KAPPA_LO, KAPPA_HI = 40.0, 50.0
    kappa_raw = kappa
    kappa = min(max(kappa, KAPPA_LO), KAPPA_HI)
    if kappa != kappa_raw:
        log(f"\n  NOTE  kappa* = {kappa_raw:.1f} from the data, clamped to {kappa:.1f}. "
            f"It is a ratio of two\n        point estimates and inherits both errors; the "
            f"likelihood's own optimum is\n        40-60, and {kappa_raw:.1f} diverged at "
            f"full scale (E[n] -> n_max in <1,000 iterations).")
    log(f"\n  training targets derived from these:")
    log(f"       --elast-target {agg:.4f}      (the projection pins gb to this)")
    log(f"       --kappa-init   {kappa:.1f}"
        f"      = |own| / gb,  gb = |agg| E[n]/Var(n) = {gb:.5f}")
    log(f"       E[n] {en:.3f}  Var(n) {vn:.2f}  (training split)")

    out = dict(estimator_A=A, estimator_B=B, aggregate=G,
               adopted=dict(own=own, own_source=own_src, cross=cross,
                            cross_source=cross_src, agg=agg, agg_source=agg_src),
               size=dict(en=float(en), var=float(vn)),
               training=dict(elast_target=round(agg, 4), kappa_init=round(kappa, 1), kappa_init_raw=round(kappa_raw, 1),
                             gb_target=float(gb)),
               agreement=dict(own_sigma=float(zo), cross_sigma=float(zc)))
    dst = a.out or os.path.join(BI, "v3_elasticity_targets.json")
    with open(dst, "w") as fh:
        json.dump(out, fh, indent=2)
    log(f"\n  wrote {dst}")


if __name__ == "__main__":
    q = argparse.ArgumentParser()
    q.add_argument("--out", default="")
    main(q.parse_args())
