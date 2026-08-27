"""Downstream evaluation: counterfactual, personalisation, segmentation, recommendation,
and segment-level generation, from one checkpoint, loaded correctly.

Two things must be right before any number here means anything, and both were wrong in the
legacy scripts:

  price parameterisation.  gamma is either the coefficient (--price-soft) or a softplus
  pre-image.  The tensors are indistinguishable, so a loader that guesses reads 0.0207 as
  softplus(0.0207) = 0.7036 -- 34x too large.  load_any now reads it from model_flags.

  polynomial degree.  The per-row ESP is truncated at m.poly_degree.  exp(-rho_c C(n,2))
  with rho_c = -0.337 and n = 120 is 10^1045, so the untruncated recursion overflows
  float64 (10^308) and pi_quad returns NaN.  Worse, degrees just below that are FINITE and
  meaningless: at degree 64 sum_j pi_j = 120.00 = n_max, i.e. every product certain, when
  the truth is 7.6.  The safe degree depends on rho_c and therefore on the checkpoint, so
  it is measured here rather than assumed -- bottom up from the largest per-category count
  actually present in the data, which is the smallest degree that can represent an observed
  basket at all.
"""
import argparse, json, os, sys
import numpy as np
import torch

from data import build
from evalall import load_any
from features import Features
from fit import Batcher
from ragged import RaggedModel


def log(s=""):
    print(s, flush=True)


def safe_degree(m, ix, floor, cand=(26, 32, 40, 48, 64, 96), tol=0.02):
    """Largest truncation degree whose E[n] still agrees with the floor's.

    sum_j pi_j = E[n] by construction, so a degree that inflates it is wrong on its face.
    Calibrating DOWNWARD from the untruncated polynomial (what fit.py does) cannot work:
    that reference is the overflowing one.  Calibrate upward from the floor instead, and
    stop at the first degree that moves the answer.
    """
    cand = sorted({int(c) for c in cand if c >= floor} | {int(floor)})
    base, chosen, table = None, int(floor), []
    for d in cand:
        m.poly_degree = d
        with torch.enable_grad():
            pi = m.pi_quad(ix)
        ok = bool(torch.isfinite(pi).all())
        en = float(pi.sum()) / ix.B if ok else float("nan")
        table.append((d, en, ok))
        if not ok:
            break
        if base is None:
            base = en
        elif abs(en - base) > tol * max(abs(base), 1e-9):
            break
        chosen = d
    m.poly_degree = chosen
    log(f"  truncation degree: floor {floor} (largest per-category count in the data), "
        f"chosen {chosen}")
    log("    " + "  ".join(f"d{d}:{('%.2f' % e) if ok else 'NaN'}" for d, e, ok in table)
        + f"   <- E[n]; the model's own E[n] is ~{base:.2f}")
    return chosen


def pick_trips(D, split, n, nmax, rng_seed=0, min_lines=2):
    lp = D["line_ptr"]
    idx = np.flatnonzero(D["trip_split"] == split)
    idx = np.array([t for t in idx if min_lines <= int(lp[t + 1]) - int(lp[t]) <= nmax])
    r = np.random.default_rng(rng_seed)
    return np.sort(r.choice(idx, size=min(n, len(idx)), replace=False))


def purchased_mask(ix, LI, LT):
    onb = torch.zeros(ix.item.shape[0], dtype=torch.bool)
    for b in range(ix.B):
        bs = set(int(x) for x in LI[LT == b])
        for i in (ix.item_trip == b).nonzero().flatten():
            if int(ix.item[i]) in bs:
                onb[int(i)] = True
    return onb



_SUBKEY = {}


def _rebar(ix, dlp, price_ref):
    """Recompute the price reference after a counterfactual price change.

    b_j = ... - gb_j [dbar + kappa (dlp_j - dbar)], so dbar is not a constant: perturbing
    dlp while holding dbar fixed silently deletes the substitution channel, which enters
    ONLY through dbar (a rival's rise moves b_j by gb(kappa-1) d_dbar, positive at kappa
    = 35.6).  Holding it fixed is what made this script report cross-price elasticities of
    -0.116 -- the pure basket-size effect -- for a model that actually substitutes.
    """
    if price_ref == "subcommodity":
        key = _SUBKEY["key"]
        num = torch.zeros(int(key.max()) + 1, dtype=dlp.dtype).index_add_(0, key, dlp)
        den = torch.zeros(int(key.max()) + 1, dtype=dlp.dtype).index_add_(
            0, key, torch.ones_like(dlp))
        return (num / den.clamp_min(1.0))[key]
    if price_ref == "category":
        num = torch.zeros(ix.n_rows, dtype=dlp.dtype).index_add_(0, ix.row_of, dlp)
        den = torch.zeros(ix.n_rows, dtype=dlp.dtype).index_add_(
            0, ix.row_of, torch.ones_like(dlp))
        return (num / den.clamp_min(1.0))[ix.row_of]
    num = torch.zeros(ix.B, dtype=dlp.dtype).index_add_(0, ix.item_trip, dlp)
    den = torch.zeros(ix.B, dtype=dlp.dtype).index_add_(
        0, ix.item_trip, torch.ones_like(dlp))
    return num / den.clamp_min(1.0)


# ---------------------------------------------------------------- 1. counterfactual
def counterfactual(m, Bt, trips, chunk, price_ref="trip"):
    """Own-price and uniform-price response, as arc elasticities.

    elasticity = d log pi_j / d log p_j, estimated as log(pi_g / pi_1) / log(g).  Reported
    at several multiples because a logit-shaped response is not constant in g.
    """
    log("\n" + "=" * 78)
    log("1. PRICE COUNTERFACTUAL")
    log("=" * 78)
    mult = np.array([1.10, 1.25, 1.50, 2.00])
    own_num = np.zeros(len(mult)); own_den = 0.0
    cross_same = np.zeros(len(mult)); cross_same_n = 0.0
    cross_diff = np.zeros(len(mult)); cross_diff_n = 0.0
    uni_en = np.zeros(len(mult)); uni_base = 0.0; nb = 0
    for k in range(0, len(trips), chunk):
        ix, ctx, lctx, hh, LI, LT, LC, LU = Bt.make(trips[k:k + chunk])
        m.house, m.ctx = hh, ctx
        if price_ref == "subcommodity":
            _SUBKEY["key"] = ix.item_trip * Bt.n_sub + Bt.sub_of[ix.item]
        onb = purchased_mask(ix, LI, LT)
        if not bool(onb.any()):
            continue
        with torch.enable_grad():
            base = m.pi_quad(ix)
        good = (base > 1e-8) & onb
        if not bool(good.any()):
            continue
        same_cat = torch.zeros_like(onb)
        for b in range(ix.B):
            cats = set(int(c) for c in ix.row_cat[ix.row_of[(ix.item_trip == b) & onb]])
            sel = (ix.item_trip == b) & (~onb)
            same_cat |= sel & torch.tensor(
                [int(ix.row_cat[ix.row_of[i]]) in cats for i in range(len(onb))])
        with torch.enable_grad():
            uni_base += float(base.sum()); nb += ix.B
            for gi, g in enumerate(mult):
                d = float(np.log(g))
                # own price: only the purchased items move
                c2 = dict(ctx); dd = ctx["dlp"].clone(); dd[onb] = dd[onb] + d
                c2["dlp"] = dd
                if "dlp_bar" in ctx:
                    c2["dlp_bar"] = _rebar(ix, dd, price_ref)
                m.ctx = c2
                p = m.pi_quad(ix)
                own_num[gi] += float((torch.log(p[good] / base[good])).sum())
                sel = same_cat & (base > 1e-8)
                if bool(sel.any()):
                    cross_same[gi] += float(torch.log(p[sel] / base[sel]).sum())
                sel2 = (~onb) & (~same_cat) & (base > 1e-8)
                if bool(sel2.any()):
                    cross_diff[gi] += float(torch.log(p[sel2] / base[sel2]).sum())
                # uniform: every price moves
                c3 = dict(ctx); c3["dlp"] = ctx["dlp"] + d
                if "dlp_bar" in ctx:
                    c3["dlp_bar"] = ctx["dlp_bar"] + d   # uniform shift: bar moves by d too
                m.ctx = c3
                uni_en[gi] += float(m.pi_quad(ix).sum())
            m.ctx = ctx
        own_den += float(good.sum())
        cross_same_n += float((same_cat & (base > 1e-8)).sum())
        cross_diff_n += float(((~onb) & (~same_cat) & (base > 1e-8)).sum())
    log(f"\n  {int(own_den):,} purchased (household, product, trip) cells\n")
    log(f"  {'price x':>9}{'own-price elast':>18}{'cross, same cat':>18}{'cross, other cat':>18}")
    for gi, g in enumerate(mult):
        lg = np.log(g)
        oe = own_num[gi] / max(own_den, 1) / lg
        cs = cross_same[gi] / max(cross_same_n, 1) / lg
        cd = cross_diff[gi] / max(cross_diff_n, 1) / lg
        log(f"  {g:>9.2f}{oe:>18.4f}{cs:>18.4f}{cd:>18.4f}")
    log(f"\n  uniform price rise (every product), E[n] per basket:")
    log(f"    baseline           {uni_base / max(nb,1):.3f}")
    for gi, g in enumerate(mult):
        e = uni_en[gi] / max(nb, 1)
        agg = np.log(e / (uni_base / max(nb, 1))) / np.log(g)
        log(f"    x{g:<5.2f}  E[n] {e:6.3f}   aggregate elasticity {agg:+.4f}")
    return dict(own_elast_10pct=float(own_num[0] / max(own_den, 1) / np.log(1.10)),
                cells=int(own_den))


# ---------------------------------------------------------------- 2. personalisation
def rank_mrr(m, Bt, trips, chunk, ablate=None):
    """MRR of purchased items ranked by pi within the trip's own assortment."""
    rr, n = 0.0, 0
    saved = {}
    if ablate:
        for name in ablate:
            p = getattr(m, name)
            saved[name] = p.detach().clone()
            with torch.no_grad():
                p.copy_(p.mean(0, keepdim=True).expand_as(p))
    try:
        for k in range(0, len(trips), chunk):
            ix, ctx, lctx, hh, LI, LT, LC, LU = Bt.make(trips[k:k + chunk])
            m.house, m.ctx = hh, ctx
            with torch.enable_grad():
                pi = m.pi_quad(ix)
            for b in range(ix.B):
                sel = (ix.item_trip == b).nonzero().flatten()
                if len(sel) == 0:
                    continue
                sc = pi[sel]
                items = ix.item[sel]
                order = torch.argsort(sc, descending=True)
                ranked = items[order]
                bought = set(int(x) for x in LI[LT == b])
                pos = {int(v): i + 1 for i, v in enumerate(ranked) if int(v) in bought}
                for it in bought:
                    if it in pos:
                        rr += 1.0 / pos[it]; n += 1
    finally:
        for name, v in saved.items():
            with torch.no_grad():
                getattr(m, name).copy_(v)
    return rr / max(n, 1), n


def personalisation(m, Bt, trips, chunk):
    log("\n" + "=" * 78)
    log("2. PERSONALISATION  (does the household actually matter?)")
    log("=" * 78)
    log("\n  Each block is replaced by its population mean, so the model keeps its")
    log("  structure but loses that source of per-household variation.  The drop is")
    log("  how much of the ranking that block was carrying.\n")
    full, n = rank_mrr(m, Bt, trips, chunk)
    log(f"  {'ablation':<34}{'MRR':>10}{'change':>12}")
    log(f"  {'none (full model)':<34}{full:>10.5f}{'--':>12}")
    out = dict(full=full, cells=n)
    for name, blocks in (("household taste theta", ["theta"]),
                         ("household price gamma", ["gamma"]),
                         ("both", ["theta", "gamma"])):
        v, _ = rank_mrr(m, Bt, trips, chunk, ablate=blocks)
        log(f"  {'-> ' + name:<34}{v:>10.5f}{100*(v-full)/max(abs(full),1e-12):>11.1f}%")
        out[name] = v
    log(f"\n  ranked over {n:,} purchased items in their own store assortments")
    return out


# ---------------------------------------------------------------- 3. segmentation
def kmeans(X, k, iters=100, seed=0):
    r = np.random.default_rng(seed)
    C = X[r.choice(len(X), k, replace=False)].copy()
    for _ in range(iters):
        d = ((X[:, None, :] - C[None, :, :]) ** 2).sum(-1)
        a = d.argmin(1)
        Cn = np.stack([X[a == j].mean(0) if (a == j).any() else C[j] for j in range(k)])
        if np.allclose(Cn, C):
            break
        C = Cn
    return a, C


def segmentation(m, D, k):
    log("\n" + "=" * 78)
    log(f"3. CUSTOMER SEGMENTATION  (k = {k}, on the model's fitted household parameters)")
    log("=" * 78)
    th = m.theta.detach().numpy()
    ga = m.price_g().detach().numpy()
    Z = (th - th.mean(0)) / (th.std(0) + 1e-9)
    lab, _ = kmeans(Z, k, seed=0)
    lp, tu = D["line_ptr"], D["trip_user"]
    sizes = np.array([int(lp[t + 1]) - int(lp[t]) for t in range(len(tu))])
    log("\n  segments are k-means on the standardised taste embedding theta [N, 32];")
    log("  price sensitivity is the fitted gamma, reported as its mean over the 8 dims.\n")
    log(f"  {'seg':>4}{'households':>12}{'mean |theta|':>14}{'mean gamma':>12}"
        f"{'trips':>9}{'obs E[n]':>10}")
    rows = []
    for j in range(k):
        who = np.flatnonzero(lab == j)
        tsel = np.isin(tu, who)
        en = sizes[tsel].mean() if tsel.any() else float("nan")
        log(f"  {j:>4}{len(who):>12}{np.abs(th[who]).mean():>14.4f}"
            f"{ga[who].mean():>12.5f}{int(tsel.sum()):>9}{en:>10.2f}")
        rows.append(dict(seg=j, n=len(who), gamma=float(ga[who].mean()), en=float(en)))
    sp = [r["gamma"] for r in rows]
    log(f"\n  price sensitivity spread across segments: {min(sp):.5f} to {max(sp):.5f} "
        f"({max(sp)/max(min(sp),1e-12):.2f}x)")
    return lab, rows


# ------------------------------------------------- 4/5. generation at segment level
def generation(m, D, Bt, lab, k, n_trips, chunk, seed=0):
    log("\n" + "=" * 78)
    log("4. GENERATION AT SEGMENT LEVEL  (sample baskets, compare to that segment's data)")
    log("=" * 78)
    log("\n  Baskets are drawn from the model itself (z -> size -> category split ->")
    log("  products).  A model that has learned the segment should reproduce its basket")
    log("  size and its top products without being told them.\n")
    lp, tu = D["line_ptr"], D["line_item"]
    trip_user = D["trip_user"]
    gen = torch.Generator().manual_seed(seed)
    items = None
    try:
        import pandas as pd
        ip = os.path.join("..", "..", "basket_input", "items.parquet")
        if os.path.exists(ip):
            items = pd.read_parquet(ip)
    except Exception:
        pass
    log(f"  {'seg':>4}{'trips':>7}{'obs E[n]':>10}{'gen E[n]':>10}{'obs var':>10}"
        f"{'gen var':>10}{'top-10 overlap':>16}")
    out = []
    for j in range(k):
        who = np.flatnonzero(lab == j)
        cand = np.flatnonzero(np.isin(trip_user, who) & (D["trip_split"] == 1))
        # UNCONDITIONAL, deliberately.  Filtering observed baskets to >= 2 lines while the
        # model samples unconditionally compares two different quantities: 18.1% of
        # validation baskets are single-line, and the filter alone raises observed E[n]
        # from 7.819 to 9.327 (+19.3%) without touching the model.  That artifact, not the
        # sampler, produced the apparent generation shortfall.
        cand = np.array([t for t in cand if 1 <= int(lp[t + 1]) - int(lp[t]) <= Bt.nmax])
        if len(cand) == 0:
            continue
        r = np.random.default_rng(100 + j)
        sel = np.sort(r.choice(cand, size=min(n_trips, len(cand)), replace=False))
        obs_sizes, obs_items = [], []
        for t in sel:
            lo, hi = int(lp[t]), int(lp[t + 1])
            obs_sizes.append(hi - lo); obs_items.extend(list(D["line_item"][lo:hi]))
        gen_sizes, gen_items = [], []
        for c in range(0, len(sel), chunk):
            ix, ctx, lctx, hh, LI, LT, LC, LU = Bt.make(sel[c:c + chunk])
            m.house, m.ctx = hh, ctx
            with torch.no_grad():
                baskets = m.sample(ix, n_draws=32, generator=gen, mode_steps=1)
            for bsk in baskets:
                gen_sizes.append(len(bsk)); gen_items.extend([int(x) for x in bsk])
        os_, gs = np.array(obs_sizes), np.array(gen_sizes)
        to = [i for i, _ in __import__("collections").Counter(obs_items).most_common(10)]
        tg = [i for i, _ in __import__("collections").Counter(gen_items).most_common(10)]
        ov = len(set(to) & set(tg))
        log(f"  {j:>4}{len(sel):>7}{os_.mean():>10.2f}{gs.mean():>10.2f}"
            f"{os_.var():>10.2f}{gs.var():>10.2f}{str(ov)+' / 10':>16}")
        out.append(dict(seg=j, obs_en=float(os_.mean()), gen_en=float(gs.mean()),
                        overlap=int(ov), top_obs=to[:5], top_gen=tg[:5]))
    log("\n  top-5 products, observed vs generated, per segment:")
    for o in out:
        log(f"    seg {o['seg']}  obs {o['top_obs']}")
        log(f"           gen {o['top_gen']}")
    return out


def main(a):
    torch.set_default_dtype(torch.float64)
    D = build()
    J, N, C, S = (int(D[k]) for k in ("n_item", "n_user", "n_cat", "n_store"))
    # The reference is a property of the CHECKPOINT; a flag cannot override what the
    # weights were fitted under without changing what they mean.
    _blob = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    _ref = str(((_blob.get("model_flags") or {}) if isinstance(_blob, dict) else {})
               .get("price_ref", a.price_ref))
    del _blob
    Bt = Batcher(D, Features(J, S, 712), a.nmax, price_ref=_ref)
    m = RaggedModel(J=J, N=N, C=C, K=32, Kz=a.Kz, nmax=a.nmax, R=a.R, S=S, Kp=8)
    meta = load_any(a.ckpt, m, J, D)
    m.double().eval()
    log("=" * 78)
    log(f"DOWNSTREAM EVALUATION  {os.path.basename(a.ckpt)}  ({meta})")
    log("=" * 78)
    lp, lc = D["line_ptr"], D["line_cat"]
    worst = 0
    for t in range(len(D["trip_user"])):
        lo, hi = int(lp[t]), int(lp[t + 1])
        if hi > lo:
            worst = max(worst, int(np.bincount(lc[lo:hi]).max()))
    trips = pick_trips(D, 1, a.n_trips, a.nmax)
    ixc, ctxc, _, hhc, _, _, _, _ = Bt.make(trips[:24])
    m.house, m.ctx = hhc, ctxc
    safe_degree(m, ixc, worst)
    res = {}
    if not a.skip_cf:
        res["counterfactual"] = counterfactual(m, Bt, trips, a.chunk, _ref)
    if not a.skip_pers:
        res["personalisation"] = personalisation(m, Bt, trips[:a.n_pers], a.chunk)
    lab, rows = segmentation(m, D, a.k)
    res["segments"] = rows
    if not a.skip_gen:
        res["generation"] = generation(m, D, Bt, lab, a.k, a.n_gen, a.chunk)
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(res, fh, indent=2, default=float)
        log(f"\nwrote {a.out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--n-trips", type=int, default=192)
    p.add_argument("--n-pers", type=int, default=96)
    p.add_argument("--n-gen", type=int, default=48)
    p.add_argument("--chunk", type=int, default=24)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--Kz", type=int, default=4)
    p.add_argument("--nmax", type=int, default=120)
    p.add_argument("--R", type=int, default=120)
    p.add_argument("--price-ref", choices=("trip","category","subcommodity"), default="trip")
    p.add_argument("--skip-cf", action="store_true")
    p.add_argument("--skip-pers", action="store_true")
    p.add_argument("--skip-gen", action="store_true")
    p.add_argument("--out", default="")
    main(p.parse_args())
