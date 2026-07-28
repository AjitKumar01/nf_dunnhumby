"""
Stage 9 -- Model-free checks that the estimated heterogeneity is real.

(a) Paper Figure 6.  For each product, split households into terciles of *predicted*
    own-price elasticity, then measure the actual Sunday->Monday change in aggregate
    demand in the held-out test set as a function of how much the price moved.  If the
    heterogeneity is spurious, the three terciles respond identically.

(b) A check the paper could not run.  dunnhumby records real targeted offers, so the
    same predicted price sensitivity can be tested against behaviour during a
    household's actual coupon-eligibility window: do the households the model calls
    price sensitive show a bigger purchase lift when they hold a coupon?

Both use only held-out data and never treat the model's own predictions as truth.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import torch

import nf_torch as nf
from importlib import import_module

trainer = import_module("05_train_nf")
ev_mod = import_module("07_evaluate")
HERE = os.path.dirname(os.path.abspath(__file__))
MI = os.path.join(HERE, "..", "model_input")
OUT = os.path.join(HERE, "..", "out")


def log(m):
    print(f"[09] {m}", flush=True)


def household_item_elasticity(m1, m2, d, dev):
    """Predicted own-price elasticity for every (household, item) at mean prices."""
    users = torch.arange(d.n_users, device=dev)
    # a representative session: the one closest to the median price level
    s_mid = int(np.argsort(d.price.mean(0).cpu().numpy())[d.n_sessions // 2])
    sess = torch.full_like(users, s_mid)
    pred = {}
    with torch.no_grad():
        chunk = 512
        acc = {k: [] for k in ["pcat", "pitem", "price", "bij", "nest"]}
        for a in range(0, d.n_users, chunk):
            uu, ss = users[a:a + chunk], sess[a:a + chunk]
            B = uu.shape[0]
            items = d.cat_items.unsqueeze(0).expand(B, -1, -1).reshape(B, -1)
            mask = d.cat_mask.unsqueeze(0).expand(B, -1, -1).reshape(B, -1)
            u = m1.utility(uu, ss, items, stoch=False).masked_fill(mask == 0, -1e9)
            u = u.reshape(B, d.n_cats, -1)
            iv = torch.logsumexp(u, 2)
            if getattr(m1, "iv_bar", None) is not None:
                iv = iv - m1.iv_bar[uu]
            acc["pcat"].append(torch.sigmoid(m2.logits(uu, ss, iv, stoch=False)).cpu())
            acc["pitem"].append((torch.softmax(u, 2) * d.cat_mask.unsqueeze(0)).cpu())
            acc["price"].append(d.price[items, ss.unsqueeze(1)].reshape(u.shape).cpu())
            b = m1.price_coefficients(uu, items).reshape(u.shape)
            acc["bij"].append(b.cpu())
            acc["nest"].append(m2.nesting_coef(uu).cpu())
        pred = {k: torch.cat(v, 0) for k, v in acc.items()}
    own = ev_mod.elasticity_tensors(pred, d.cat_mask.cpu().unsqueeze(0))[0]
    # -> long frame (user_id, item_id, elasticity)
    ci = d.cat_items.cpu().numpy()
    cm = d.cat_mask.cpu().numpy()
    rows = []
    for c in range(ci.shape[0]):
        for s in np.where(cm[c] > 0)[0]:
            rows.append(pd.DataFrame({"user_id": np.arange(d.n_users),
                                      "item_id": int(ci[c, s]),
                                      "elasticity": own[:, c, s].numpy()}))
    return pd.concat(rows, ignore_index=True)


def demand_change_by_tercile(el, d):
    """Paper Figure 6: realised Sunday->Monday demand change by price-change bucket
    and predicted-elasticity tercile, in the test set."""
    obs = pd.read_csv(os.path.join(MI, "id_maps", "observations.csv"))
    ev = pd.read_csv(os.path.join(MI, "events.csv"))

    el = el.copy()
    # elasticities are negative, so the smallest values are the most elastic
    el["tercile"] = el.groupby("item_id").elasticity.transform(
        lambda v: pd.qcut(v.rank(method="first"), 3,
                          labels=["most elastic", "middle", "least elastic"]))

    test = obs[obs.split == "test"]
    trips = test[["user_id", "session_id", "pair_week", "weekday"]].drop_duplicates(
        ["user_id", "session_id"])

    out = []
    for item, g in el.groupby("item_id"):
        tmap = g.set_index("user_id").tercile
        tr = trips.assign(tercile=trips.user_id.map(tmap)).dropna(subset=["tercile"])
        denom = tr.groupby(["pair_week", "weekday", "tercile"], observed=True).size()
        bj = test[test.item_id == item]
        bj = bj.assign(tercile=bj.user_id.map(tmap)).dropna(subset=["tercile"])
        num = bj.groupby(["pair_week", "weekday", "tercile"], observed=True).size()
        df = pd.concat([denom.rename("trips"), num.rename("buys")], axis=1)
        df = df.fillna({"buys": 0}).reset_index()
        df = df[df.trips > 0]
        w = df.pivot_table(index=["pair_week", "tercile"], columns="weekday",
                           values=["buys", "trips"], observed=True)
        if w.empty or ("buys", 0) not in w.columns or ("buys", 1) not in w.columns:
            continue
        w = w.dropna()
        if w.empty:
            continue
        d_rate = w[("buys", 1)] / w[("trips", 1)] - w[("buys", 0)] / w[("trips", 0)]
        r = d_rate.rename("d_rate").reset_index()
        r["item_id"] = item
        r["n"] = (w[("trips", 0)] + w[("trips", 1)]).values
        out.append(r)
    if not out:
        return None
    res = pd.concat(out, ignore_index=True).dropna(subset=["d_rate"])
    res = res.merge(ev[["item_id", "pair_week", "dp"]], on=["item_id", "pair_week"], how="left")
    bins = [-np.inf, -0.25, -0.10, -0.01, 0.01, 0.10, 0.25, np.inf]
    labels = ["> .25 cut", ".10-.25 cut", ".01-.10 cut", "no change",
              ".01-.10 rise", ".10-.25 rise", "> .25 rise"]
    res["bucket"] = pd.cut(res.dp, bins, labels=labels)
    tab = res.groupby(["bucket", "tercile"], observed=True).apply(
        lambda g: pd.Series({"demand_change": np.average(g.d_rate, weights=g.n) * 1000,
                             "cells": len(g), "trips": g.n.sum()}), include_groups=False)
    return tab.reset_index()


def coupon_response(el, d):
    """Purchase lift while holding a coupon, by predicted price-sensitivity tercile."""
    npz = os.path.join(MI, "coupon_campaigns.npz")
    if not os.path.exists(npz):
        return None
    z = np.load(npz)
    U, P, S, W = z["U"], z["P"], z["S"], z["w"]
    obs = pd.read_csv(os.path.join(MI, "id_maps", "observations.csv"))
    trips = obs[["user_id", "session_id", "split"]].drop_duplicates(["user_id", "session_id"])

    el = el.copy()
    el["tercile"] = el.groupby("item_id").elasticity.transform(
        lambda s: pd.qcut(s.rank(method="first"), 3,
                          labels=["most elastic", "middle", "least elastic"]))
    tmap = el.set_index(["user_id", "item_id"]).tercile

    rows = []
    n_camp = U.shape[1]
    for k in range(n_camp):
        us = np.where(U[:, k] > 0)[0]
        js = np.where(P[:, k] > 0)[0]
        ss = np.where(S[:, k] > 0)[0]
        if len(us) == 0 or len(js) == 0 or len(ss) == 0:
            continue
        in_win = trips[trips.user_id.isin(us) & trips.session_id.isin(ss)]
        out_win = trips[trips.user_id.isin(us) & ~trips.session_id.isin(ss)]
        buys = obs[obs.user_id.isin(us) & obs.item_id.isin(js)]
        for j in js:
            bj = buys[buys.item_id == j]
            b_in = bj[bj.session_id.isin(ss)]
            b_out = bj[~bj.session_id.isin(ss)]
            if len(in_win) == 0 or len(out_win) == 0:
                continue
            for label in ["most elastic", "middle", "least elastic"]:
                sel = {u for u in us if tmap.get((u, j)) == label}
                if not sel:
                    continue
                ni = in_win.user_id.isin(sel).sum()
                no = out_win.user_id.isin(sel).sum()
                if ni == 0 or no == 0:
                    continue
                rows.append({"campaign": k, "certain": bool(W[k] >= 0.999),
                             "item_id": int(j), "tercile": label,
                             "rate_in": b_in.user_id.isin(sel).sum() / ni,
                             "rate_out": b_out.user_id.isin(sel).sum() / no,
                             "trips_in": ni, "trips_out": no})
    if not rows:
        return None
    df = pd.DataFrame(rows)
    df["lift"] = df.rate_in - df.rate_out

    def summarise(g):
        return pd.Series({
            "rate_with_coupon": np.average(g.rate_in, weights=g.trips_in),
            "rate_without": np.average(g.rate_out, weights=g.trips_out),
            "lift": np.average(g.lift, weights=g.trips_in),
            "cells": len(g)})

    tabs = []
    for name, sub in [("all campaigns", df), ("certain eligibility (TypeB/C)",
                                              df[df.certain])]:
        if not len(sub):
            continue
        t = sub.groupby("tercile", observed=True).apply(summarise, include_groups=False)
        t["lift_pct"] = 100 * t.lift / t.rate_without
        t["sample"] = name
        tabs.append(t.reset_index())
    return pd.concat(tabs, ignore_index=True) if tabs else None


def main(a):
    dev = trainer.pick_device(a.device)
    extras = json.load(open(os.path.join(OUT, f"{a.label}_history.json")))["config"]["extras"]
    d = nf.load(MI, device=dev, extras=extras)
    m1, m2, _ = ev_mod.load_model(a.label, d, dev)
    log(f"model {a.label} on {dev}")

    el = household_item_elasticity(m1, m2, d, dev)
    el.to_parquet(os.path.join(OUT, f"{a.label}_elasticities.parquet"), index=False)
    log(f"household x item elasticities: {len(el):,}  median {el.elasticity.median():.3f}")

    tab = demand_change_by_tercile(el, d)
    if tab is not None:
        tab.to_csv(os.path.join(OUT, f"{a.label}_demand_by_tercile.csv"), index=False)
        piv = tab.pivot(index="bucket", columns="tercile", values="demand_change")
        print("\nRealised Sun->Mon demand change per 1,000 trips, test set "
              "(paper Figure 6)\n" + piv.round(3).to_string())

    cr = coupon_response(el, d)
    if cr is not None:
        cr.to_csv(os.path.join(OUT, f"{a.label}_coupon_response.csv"), index=False)
        print("\nPurchase rate while holding a coupon, by predicted elasticity tercile\n"
              + cr.round(5).to_string(index=False))
    log("done")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--label", default="nf")
    p.add_argument("--device", default="cpu")
    main(p.parse_args())
