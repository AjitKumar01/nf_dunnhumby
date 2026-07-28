"""
Stage 7 -- Reproduce the paper's evaluation battery on dunnhumby.

  sec. 6.1  predictive fit overall and by category         (Tables 1, 2)
  sec. 6.2  fit in weeks with a price-change event         (Table 3)
  sec. 6.3  degree of personalisation, and predictions for
            households that never bought the item          (Table 4, Figure 5)
  sec. 6.4  own- and cross-price elasticities, and whether
            cross-price elasticities are higher inside a
            sub-commodity than across it                   (Tables 5, 7)

Unconditional purchase probability, as in the paper:
    P(choose j on trip t) = P(buy from category c) * P(j | buy from c)

Usage:  python 07_evaluate.py --labels nf logit nf_promo
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
HERE = os.path.dirname(os.path.abspath(__file__))
MI = os.path.join(HERE, "..", "model_input")
OUT = os.path.join(HERE, "..", "out")
EPS = 1e-12


def log(m):
    print(f"[07] {m}", flush=True)


# --------------------------------------------------------------------- loading
def load_model(label, d, dev):
    cfg = json.load(open(os.path.join(OUT, f"{label}_history.json")))["config"]
    m1 = nf.ProductChoice(d, K=cfg["K"], Kp=cfg["Kp"], use_user_obs=not cfg["no_user_obs"],
                          use_item_obs=cfg["item_obs"], extras=cfg["extras"],
                          homogeneous=cfg["homogeneous"], prior_var=cfg["prior_var"],
                          intercept_var=cfg.get("intercept_var"),
                          price_prior_var=cfg.get("price_prior_var"),
                          price_prior_mean=cfg.get("price_prior_mean", 0.5),
                          scale_prior=not cfg.get("no_scale_prior", False),
                          pool_across_categories=not cfg.get("no_pool", False)).to(dev)
    m1.load_state_dict(torch.load(os.path.join(OUT, f"{label}_stage1.pt"), map_location=dev))
    m2 = nf.CategoryChoice(d, K=cfg["K2"], Kiv=cfg["Kiv"], Ktime=cfg["Ktime"],
                           use_user_obs=cfg["cat_user_obs"], homogeneous=cfg["homogeneous"],
                           prior_var=cfg["prior_var"],
                           scale_prior=not cfg.get("no_scale_prior", False)).to(dev)
    m2.load_state_dict(torch.load(os.path.join(OUT, f"{label}_stage2.pt"), map_location=dev))
    m1.eval(); m2.eval()
    # stage 2 was trained on centred inclusive values, so predictions must centre too
    m1.iv_bar = None if cfg.get("no_center_iv") else m1.mean_inclusive_values()
    return m1, m2, cfg


@torch.no_grad()
def trip_predictions(m1, m2, d, split, chunk=1024):
    """Per trip: P(category), P(item | category), price, price coefficient, nesting."""
    tu, ts = d.trips[split]
    T, C, M = tu.shape[0], d.n_cats, d.cat_items.shape[1]
    outs = {k: [] for k in ["pcat", "pitem", "price", "bij", "nest"]}
    for a in range(0, T, chunk):
        uu, ss = tu[a:a + chunk], ts[a:a + chunk]
        B = uu.shape[0]
        items = d.cat_items.unsqueeze(0).expand(B, -1, -1).reshape(B, -1)
        mask = d.cat_mask.unsqueeze(0).expand(B, -1, -1).reshape(B, -1)
        u = m1.utility(uu, ss, items, stoch=False).masked_fill(mask == 0, -1e9)
        u = u.reshape(B, C, M)
        iv = torch.logsumexp(u, dim=2)
        if getattr(m1, "iv_bar", None) is not None:
            iv = iv - m1.iv_bar[uu]
        pitem = torch.softmax(u, dim=2) * d.cat_mask.unsqueeze(0)
        outs["pcat"].append(torch.sigmoid(m2.logits(uu, ss, iv, stoch=False)).cpu())
        outs["pitem"].append(pitem.cpu())
        outs["price"].append(d.price[items, ss.unsqueeze(1)].reshape(B, C, M).cpu())
        b = m1.price_coefficients(uu, items).reshape(B, C, M)
        outs["bij"].append(b.cpu())
        outs["nest"].append(m2.nesting_coef(uu).cpu())
    return {k: torch.cat(v, 0) for k, v in outs.items()}


def slot_lookup(d):
    """(category, item) -> slot in the padded item block."""
    ci = d.cat_items.cpu().numpy()
    cm = d.cat_mask.cpu().numpy()
    return {(c, int(ci[c, s])): s for c in range(ci.shape[0])
            for s in range(ci.shape[1]) if cm[c, s] > 0}


def outcome_matrix(d, split, slots):
    """[T, C, M] indicator of the chosen item, plus the trip index frame."""
    tu, ts = d.trips[split]
    T, C, M = tu.shape[0], d.n_cats, d.cat_items.shape[1]
    key = (tu.to(torch.int64) * d.n_sessions + ts.to(torch.int64)).cpu().numpy()
    pos = {int(k): i for i, k in enumerate(key)}
    ou, oi, os_ = d.obs[split]
    ok = (ou.to(torch.int64) * d.n_sessions + os_.to(torch.int64)).cpu().numpy()
    rows = np.fromiter((pos[int(k)] for k in ok), dtype=np.int64, count=len(ok))
    cats = d.item_cat[oi].cpu().numpy()
    slot = np.fromiter((slots[(int(c), int(j))] for c, j in zip(cats, oi.cpu().numpy())),
                       dtype=np.int64, count=len(rows))
    y = torch.zeros((T, C, M))
    y[rows, cats, slot] = 1.0
    return y


# ------------------------------------------------------------------ sec 6.1/6.2
def predictive_fit(pred, y, mask):
    p = (pred["pcat"].unsqueeze(2) * pred["pitem"]).clamp(EPS, 1 - EPS)
    n = float(y.sum())
    ll = float((y * torch.log(p)).sum()) / n
    se = float((((y - p) ** 2) * mask).sum()) / n
    return ll, se, n, p


def event_fit(p, y, ev_mask):
    n = float((y * ev_mask).sum())
    if n == 0:
        return np.nan, 0
    return float((y * ev_mask * torch.log(p)).sum()) / n, int(n)


# --------------------------------------------------------------------- sec 6.3
def household_rates(p, y, users, valid):
    """Average predicted and realised purchase rate per (household, item)."""
    T = p.shape[0]
    pm = p.reshape(T, -1).numpy()[:, valid]
    ym = y.reshape(T, -1).numpy()[:, valid]
    uniq, inv = np.unique(users, return_inverse=True)
    U = len(uniq)
    cnt = np.bincount(inv, minlength=U).astype(float)[:, None]
    pred = np.zeros((U, pm.shape[1])); act = np.zeros((U, pm.shape[1]))
    np.add.at(pred, inv, pm)
    np.add.at(act, inv, ym)
    return uniq, pred / cnt, act / cnt


def slope_with_item_fe(pred, act):
    """Regress realised on predicted rate with item fixed effects (paper Table 4)."""
    pm = pred - pred.mean(0, keepdims=True)
    am = act - act.mean(0, keepdims=True)
    den = float((pm * pm).sum())
    return float((pm * am).sum()) / den if den > 0 else np.nan


def never_buyer_curve(pred, act, train_counts, n_dec=10):
    """Realised held-out rate by decile of predicted rate, among household-item
    pairs with zero training purchases (paper Figure 5)."""
    m = train_counts == 0
    if m.sum() < n_dec * 10:
        return None
    pv, av = pred[m], act[m]
    q = pd.qcut(pd.Series(pv).rank(method="first"), n_dec, labels=False)
    df = pd.DataFrame({"decile": q, "actual": av, "predicted": pv})
    g = df.groupby("decile").agg(actual=("actual", "mean"), predicted=("predicted", "mean"),
                                 n=("actual", "size")).reset_index()
    g["lift_vs_bottom"] = g.actual / max(g.actual.iloc[0], EPS)
    return g


# --------------------------------------------------------------------- sec 6.4
def elasticity_tensors(pred, mask):
    """Household-level own- and cross-price elasticities.

    With P(j) = P(c) * P(j|c) and u_ijt containing -b_ij * p_jt:
        own_ij   = -p_j b_ij [ (1 - P_j|c) + nest_ic * P_j|c * (1 - P_c) ]
        cross_kj = +p_j b_ij P_j|c [ 1 - nest_ic * (1 - P_c) ]
    The cross term is the same for every k in the category at the *household*
    level -- that is the IIA restriction.  What breaks it in aggregate is the
    weighting: households who are likely to buy k get more weight in k's
    aggregate cross elasticity, so similar products end up more substitutable.
    """
    pcat, pitem, price, bij, nest = (pred[k] for k in ["pcat", "pitem", "price", "bij", "nest"])
    nest3 = nest.unsqueeze(2)
    own = -(price * bij) * ((1 - pitem) + nest3 * pitem * (1 - pcat).unsqueeze(2))
    cross_j = (price * bij) * pitem * (1 - nest3 * (1 - pcat).unsqueeze(2))
    # within-category (conditional-on-purchase) substitution: d log P(k|c) / d log p_j.
    # This is the substitution the nest is meant to describe, and unlike the
    # unconditional version it does not pass through zero when the nesting
    # coefficient sits at 1.
    cross_cond_j = (price * bij) * pitem
    return own * mask, cross_j * mask, cross_cond_j * mask


def aggregate_cross(pred, cross_j, d, items_meta, weight="unconditional", prefix="cross"):
    """Aggregate cross elasticity of item k with respect to the price of item j,
    weighted by each household's probability of buying k, then split by whether
    j and k are similar along a field the model never sees (paper sec. 6.4.1).

    At the household level the cross elasticity does not depend on k -- that is
    IIA.  What can break it in aggregate is the weighting: households likely to buy
    k get more weight in k's cross elasticity, so if the model has learned that
    similar products appeal to the same households, similar products end up more
    substitutable.  Levels differ hugely across categories, so the comparison is
    made *within* category and then averaged, rather than pooling all pairs.
    """
    pk = (pred["pcat"].unsqueeze(2) * pred["pitem"]) if weight == "unconditional" \
        else pred["pitem"]                                    # [T, C, M]
    ci = d.cat_items.cpu().numpy()
    cm = d.cat_mask.cpu().numpy()
    meta = items_meta.set_index("item_id")
    fields = {"subclass": meta.SUB_COMMODITY_DESC.to_dict(),
              "manufacturer": meta.MANUFACTURER.to_dict(),
              "brandtype": meta.BRAND.to_dict()}
    per_cat = {f: [] for f in fields}
    pooled = {f: ([], []) for f in fields}

    for c in range(ci.shape[0]):
        keep = np.where(cm[c] > 0)[0]
        if len(keep) < 2:
            continue
        P = pk[:, c, keep]                    # [T, m]
        E = cross_j[:, c, keep]               # [T, m]
        agg = (P.T @ E).numpy() / np.maximum(P.sum(0).numpy()[:, None], EPS)  # [k, j]
        for fname, lut in fields.items():
            ins, outs = [], []
            for a, sa in enumerate(keep):
                for b_, sb in enumerate(keep):
                    if a == b_:
                        continue
                    k, j = int(ci[c, sa]), int(ci[c, sb])
                    v = float(agg[a, b_])
                    (ins if lut.get(j) == lut.get(k) else outs).append(v)
            pooled[fname][0].extend(ins)
            pooled[fname][1].extend(outs)
            if ins and outs:
                per_cat[fname].append((float(np.mean(ins)), float(np.mean(outs))))

    out = {}
    for fname in fields:
        ins, outs = pooled[fname]
        mi = float(np.mean(ins)) if ins else np.nan
        mo = float(np.mean(outs)) if outs else np.nan
        pc = per_cat[fname]
        out[f"{prefix}_same_{fname}"] = mi
        out[f"{prefix}_diff_{fname}"] = mo
        out[f"{prefix}_gap_{fname}"] = float(np.mean([a - b for a, b in pc])) if pc else np.nan
        out[f"{prefix}_share_cats_higher_inside_{fname}"] = (
            float(np.mean([a > b for a, b in pc])) if pc else np.nan)
        out[f"{prefix}_n_cats_{fname}"] = len(pc)
    return out


# ---------------------------------------------------------------------- masks
def build_event_masks(d):
    """[T, C, M] masks flagging item-weeks with an own- or cross-price change."""
    ev = pd.read_csv(os.path.join(MI, "events.csv"))
    sess = pd.read_csv(os.path.join(MI, "id_maps", "sessions.csv"))
    _, ts = d.trips["test"]
    pw = sess.set_index("session_id").pair_week.reindex(ts.cpu().numpy()).values
    ci = d.cat_items.cpu().numpy()
    C, M = ci.shape
    weeks = np.unique(pw)
    wpos = {w: i for i, w in enumerate(weeks)}
    trip_w = np.array([wpos[w] for w in pw])
    out = {}
    for col in ["own_price_change", "cross_price_change"]:
        lut = np.zeros((len(weeks), C, M), dtype=bool)
        sel = ev[ev[col] == 1]
        s = {(int(i), int(w)) for i, w in zip(sel.item_id, sel.pair_week)}
        for c in range(C):
            for slot in range(M):
                j = int(ci[c, slot])
                for w in weeks:
                    if (j, w) in s:
                        lut[wpos[w], c, slot] = True
        out[col] = torch.as_tensor(lut[trip_w])
    return out


def train_purchase_counts(d, slots):
    ou, oi, _ = d.obs["train"]
    U, C, M = d.n_users, d.n_cats, d.cat_items.shape[1]
    counts = np.zeros((U, C * M))
    cats = d.item_cat[oi].cpu().numpy()
    for u, c, j in zip(ou.cpu().numpy(), cats, oi.cpu().numpy()):
        counts[u, int(c) * M + slots[(int(c), int(j))]] += 1
    return counts


# ---------------------------------------------------------------------- runner
def evaluate(label, d, dev, items_meta, ev_masks, train_counts, slots):
    m1, m2, cfg = load_model(label, d, dev)
    mask = d.cat_mask.cpu().unsqueeze(0)                # [1, C, M]
    valid = d.cat_mask.cpu().reshape(-1).numpy() > 0
    res = {"label": label,
           "K": cfg["K"], "Kp": cfg["Kp"], "homogeneous": cfg["homogeneous"],
           "pooled": not cfg.get("no_pool", False),
           "extras": ",".join(cfg["extras"]) or "-"}

    for split in ["train", "test"]:
        pred = trip_predictions(m1, m2, d, split)
        y = outcome_matrix(d, split, slots)
        ll, se, npur, p = predictive_fit(pred, y, mask)
        res[f"{split}_loglik"], res[f"{split}_mse"] = ll, se
        res[f"{split}_purchases"] = npur
        if split != "test":
            continue

        # ---- per category (paper Table 2)
        per_cat = []
        for c in range(d.n_cats):
            n = float(y[:, c].sum())
            if n == 0:
                continue
            per_cat.append({"group_id": c,
                            "loglik": float((y[:, c] * torch.log(p[:, c])).sum()) / n,
                            "mse": float((((y[:, c] - p[:, c]) ** 2) * mask[:, c]).sum()) / n,
                            "purchases": n})
        res["per_category"] = per_cat

        # ---- counterfactual events (paper Table 3)
        for name, em in ev_masks.items():
            v, n = event_fit(p, y, em)
            res[f"event_{name}_loglik"], res[f"event_{name}_n"] = v, n

        # ---- personalisation (paper Table 4) and never-buyers (Figure 5)
        tu, _ = d.trips["test"]
        uniq, pred_u, act_u = household_rates(p, y, tu.cpu().numpy(), valid)
        res["cv_upc"] = float(np.nanmean(pred_u.std(0) / np.maximum(pred_u.mean(0), EPS)))
        C, M = d.n_cats, d.cat_items.shape[1]
        slot_cat = np.repeat(np.arange(C), M)[valid]
        pc = np.stack([pred_u[:, slot_cat == c].sum(1) for c in range(C)], 1)
        ac = np.stack([act_u[:, slot_cat == c].sum(1) for c in range(C)], 1)
        res["cv_category"] = float(np.nanmean(pc.std(0) / np.maximum(pc.mean(0), EPS)))
        res["slope_upc"] = slope_with_item_fe(pred_u, act_u)
        res["slope_category"] = slope_with_item_fe(pc, ac)

        tc = train_counts[uniq][:, valid]
        nb = never_buyer_curve(pred_u.ravel(), act_u.ravel(), tc.ravel())
        res["never_buyer_upc"] = None if nb is None else nb.to_dict("records")
        tc_cat = np.stack([train_counts[uniq][:, valid][:, slot_cat == c].sum(1)
                           for c in range(C)], 1)
        nbc = never_buyer_curve(pc.ravel(), ac.ravel(), tc_cat.ravel())
        res["never_buyer_category"] = None if nbc is None else nbc.to_dict("records")

        # ---- elasticities (paper Tables 5, 7)
        own, cross_j, cross_cond_j = elasticity_tensors(pred, mask)
        mflat = (mask.expand_as(own) > 0)
        res["own_elasticity_median"] = float(own[mflat].median())
        item_mean = (own.sum(0) / own.shape[0])[d.cat_mask.cpu() > 0]
        res["own_elasticity_sd_across_items"] = float(item_mean.std())
        res["own_elasticity_mean_sd_across_users"] = float(
            own.std(0)[d.cat_mask.cpu() > 0].mean())
        res["nesting_coef_mean"] = float(pred["nest"].mean())
        res.update(aggregate_cross(pred, cross_j, d, items_meta,
                                   weight="unconditional", prefix="cross"))
        res.update(aggregate_cross(pred, cross_cond_j, d, items_meta,
                                   weight="conditional", prefix="crosscond"))
    return res


def main(a):
    global MI
    if a.indir:
        MI = os.path.join(HERE, "..", a.indir)
        trainer.set_indir(MI)
    dev = trainer.pick_device(a.device)
    items_meta = pd.read_csv(os.path.join(MI, "id_maps", "items.csv"))
    extras = sorted({e for lb in a.labels for e in
                     json.load(open(os.path.join(OUT, f"{lb}_history.json")))["config"]["extras"]})
    d = nf.load(MI, device=dev, extras=extras)
    log(f"device {dev};  extras loaded: {extras or '-'}")
    slots = slot_lookup(d)
    ev_masks = build_event_masks(d)
    tc = train_purchase_counts(d, slots)

    results = []
    for lb in a.labels:
        log(f"evaluating {lb} ...")
        results.append(evaluate(lb, d, dev, items_meta, ev_masks, tc, slots))
    with open(os.path.join(OUT, f"evaluation{a.tag}.json"), "w") as f:
        json.dump(results, f, indent=2, default=float)

    cols = ["label", "K", "Kp", "homogeneous", "pooled", "extras",
            "train_loglik", "test_loglik", "train_mse", "test_mse",
            "event_own_price_change_loglik", "event_cross_price_change_loglik",
            "cv_upc", "cv_category", "slope_upc", "slope_category",
            "own_elasticity_median", "own_elasticity_sd_across_items",
            "own_elasticity_mean_sd_across_users", "nesting_coef_mean",
            "crosscond_same_subclass", "crosscond_diff_subclass",
            "crosscond_gap_subclass", "crosscond_share_cats_higher_inside_subclass",
            "crosscond_gap_manufacturer", "crosscond_share_cats_higher_inside_manufacturer",
            "cross_gap_subclass", "cross_share_cats_higher_inside_subclass"]
    tbl = pd.DataFrame(results)[cols]
    tbl.to_csv(os.path.join(OUT, f"evaluation_summary{a.tag}.csv"), index=False)
    pd.set_option("display.width", 300)
    print("\n" + tbl.round(4).to_string(index=False))

    pc = pd.DataFrame([{"label": r["label"], **row} for r in results for row in r["per_category"]])
    if pc.label.nunique() > 1:
        pc["rank_ll"] = pc.groupby("group_id").loglik.rank(ascending=False)
        pc["rank_se"] = pc.groupby("group_id").mse.rank(ascending=True)
        rk = pc.groupby("label").agg(mean_rank_ll=("rank_ll", "mean"),
                                     mean_rank_se=("rank_se", "mean"),
                                     best_ll=("rank_ll", lambda s: (s == 1).mean()),
                                     best_se=("rank_se", lambda s: (s == 1).mean()))
        rk.to_csv(os.path.join(OUT, f"evaluation_by_category{a.tag}.csv"))
        print("\nper-category ranking (paper Table 2)\n" + rk.round(3).to_string())
    pc.to_csv(os.path.join(OUT, f"evaluation_per_category_raw{a.tag}.csv"), index=False)

    for r in results:
        if r.get("never_buyer_upc"):
            nb = pd.DataFrame(r["never_buyer_upc"])
            print(f"\nnever-bought-this-UPC deciles, {r['label']} "
                  f"(top/bottom lift {nb.actual.iloc[-1]/max(nb.actual.iloc[0],EPS):.1f}x)")
            print(nb[["decile", "predicted", "actual", "n"]].round(6).to_string(index=False))
    log("wrote out/evaluation.json, evaluation_summary.csv, evaluation_by_category.csv")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--labels", nargs="+", default=["nf"])
    p.add_argument("--indir", default="", help="model input directory (default model_input)")
    p.add_argument("--tag", default="", help="suffix for the output files")
    # cpu, not auto: on Apple silicon "auto" selects MPS, whose float32
    # reductions differ enough to move the reported log-likelihoods.  Pass
    # --device mps or --device cuda explicitly to opt in.
    p.add_argument("--device", default="cpu")
    main(p.parse_args())
