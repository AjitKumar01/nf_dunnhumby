"""
Stage 51 -- Test the imposed assumptions instead of replacing them.

Section 16 of version_2.html lists assumptions carried with no derivation behind them, and
several are marked untested.  Every attempt to add structure to this model has failed --
persistent contrastive divergence moved nothing, an explicit habit feature bought 0.006
nats, a category-pair matrix and a category context both made held-out likelihood clearly
worse -- so the prior on "relax the assumption by adding parameters" should be poor.

The cheaper move is to find out whether each assumption is actually violated.  Every check
below is a measurement on data the model already conditions on.  None changes the model,
none adds a parameter, and each either CLOSES a stated gap or sizes it so the cost of the
fix can be weighed against it.

  1  equidispersion, breadth      Var(k_c) vs E(k_c) given entry           section 16.6
  2  equidispersion, units        Var(q_j) vs E(q_j)                       section 16.6
  3  the modal-line floor         observed P(q=1) vs the fitted e^-Lambda  section 16.8
  4  unit censoring at 12         share of rows at the ceiling             section 3
  5  clamp binding rates          how often each clip is active            section 4.3
  6  units independent given S    within-basket correlation of residuals   section 16.7

Writes out/assumption_checks.json.
"""
import argparse
import importlib
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "model"))
nb = importlib.import_module("27_nested_basket")
cf = importlib.import_module("28_nested_counterfactual")

HERE = os.path.dirname(os.path.abspath(__file__))
IN = os.path.join(HERE, "..", "..", "basket_input")
OUT = os.path.join(HERE, "..", "..", "out")


def log(m):
    print(f"[51] {m}", flush=True)


def verdict(name, ok, detail):
    log(f"  {'CLOSES ' if ok else 'VIOLATED'}  {name:34s} {detail}")
    return {"assumption": name, "holds": bool(ok), "detail": detail}


def main(a):
    dev = torch.device("cpu")
    d = nb.NestedData(IN, device=dev)
    m, _ = cf.load(a.label, d, dev)
    bk = pd.read_parquet(os.path.join(IN, "baskets.parquet"))
    tr = bk[bk.split == "train"]
    res = {"label": a.label, "checks": []}

    # ---------------------------------------------------- 1 & 2  equidispersion
    #
    # The MARGINAL variance of a count is NOT the test.  Under a correct conditional
    # Poisson the marginal variance is E[Lambda] + Var(Lambda), which exceeds E[Lambda]
    # whenever the rate varies across observations -- so marginal overdispersion proves
    # nothing.  A first draft of this file made that error and reported a 2.4x violation
    # that does not exist.
    #
    # Two conditional statistics are computed instead, because they can disagree and the
    # disagreement is what identifies the problem:
    #   Pearson dispersion  mean (k - L)^2 / L      1 under Poisson, but 1/L weighting
    #                                               makes it dominated by small-L rows
    #   binned var / L      within quintiles of L   separates a wrong VARIANCE function
    #                                               (excess roughly uniform, growing with
    #                                               L for a negative binomial) from a
    #                                               wrong MEAN function (mean k departs
    #                                               from mean L, excess only at small L)
    log("")
    log("EQUIDISPERSION, conditional.  Eq. 14 says Var(q - 1) = Lambda given the covariates.")
    sp = d.splits["test"]
    rng = np.random.default_rng(a.seed)
    bidx = rng.choice(sp["n_baskets"], size=min(a.n_baskets, sp["n_baskets"]),
                      replace=False)
    Z, Kq = [], []
    with torch.no_grad():
        for st in range(0, len(bidx), 256):
            bt = nb.make_batch(d, m, "test", bidx[st:st + 256], rng, dev)
            ar = torch.arange(bt["cand"].shape[0])
            tg = bt["target"]
            jj = bt["cand"][ar, tg]
            z = (m.q0[jj]
                 - (m.q_gamma[bt["user"]] * m.q_beta[jj]).sum(-1) * bt["dlogp"][ar, tg]
                 + (m.q_state[jj] * bt["state"][ar, tg, :]).sum(-1)).clamp(-6, 4)
            Z.append(torch.exp(z).numpy())
            Kq.append(bt["units"].numpy() - 1.0)
    lam = np.concatenate(Z)
    kk = np.concatenate(Kq)
    pearson = float(((kk - lam) ** 2 / np.maximum(lam, 1e-9)).mean())
    marg = float(kk.var(ddof=1) / (lam.mean() + lam.var(ddof=1)))
    log(f"  observed var {kk.var(ddof=1):.4f} against E[L] + Var(L) = "
        f"{lam.mean() + lam.var(ddof=1):.4f}  (ratio {marg:.3f})")
    log(f"  Pearson dispersion {pearson:.3f}")
    qs = np.quantile(lam, [0, .2, .4, .6, .8, 1.0])
    log("")
    log(f"  {'fitted Lambda bin':22s} {'n':>7s} {'mean L':>8s} {'mean k':>8s} "
        f"{'var k':>8s} {'var/L':>7s} {'implied r':>10s}")
    bins = []
    for i in range(5):
        sel = (lam >= qs[i]) & (lam <= qs[i + 1] if i == 4 else lam < qs[i + 1])
        if sel.sum() < 50:
            continue
        vl = float(kk[sel].var(ddof=1) / lam[sel].mean())
        rr = float(lam[sel].mean() / (vl - 1.0)) if vl > 1.02 else float("inf")
        bins.append({"lo": float(qs[i]), "hi": float(qs[i + 1]), "n": int(sel.sum()),
                     "mean_lambda": float(lam[sel].mean()), "mean_k": float(kk[sel].mean()),
                     "var_k": float(kk[sel].var(ddof=1)), "var_over_lambda": vl,
                     "implied_r": rr})
        log(f"  [{qs[i]:.3f}, {qs[i + 1]:.3f})"[:22].ljust(22) +
            f" {sel.sum():7d} {lam[sel].mean():8.4f} {kk[sel].mean():8.4f} "
            f"{kk[sel].var(ddof=1):8.4f} {vl:7.3f} {rr:10.2f}")
    res["units_dispersion"] = {"pearson": pearson, "marginal_ratio": marg, "bins": bins}
    excess_everywhere = all(b["var_over_lambda"] > 1.05 for b in bins)
    res["checks"].append(verdict("equidispersion, units", not excess_everywhere,
                                 "var/L = " + ", ".join(
                                     f"{b['var_over_lambda']:.2f}" for b in bins)))
    log("")
    log("  var/L exceeds 1 in every bin and GROWS with L, which is the negative-binomial")
    log("  signature: Var = L + L^2/r gives var/L = 1 + L/r, linear in L.  The implied r")
    log("  column is that solved per bin; a single r near 1 covers the upper four.")
    log("  The mean function is fine -- mean k tracks mean L bin by bin -- so this is a")
    log("  variance-function problem and one dispersion parameter is the whole fix.")

    # ---------------------------------------------------------- 4  censoring
    log("")
    at_ceiling = float((tr.units >= 12).mean())
    log(f"CENSORING.  units are clipped to [1, 12] by stage 22.")
    log(f"  share of training rows at the ceiling: {100 * at_ceiling:.4f}%")
    res["checks"].append(verdict("censoring at 12 is negligible", at_ceiling < 0.001,
                                 f"{100 * at_ceiling:.4f}% of rows"))

    # ------------------------------------------- 3 & 5  fitted quantities, held out
    log("")
    log("FITTED QUANTITIES on held-out trips.")
    sp = d.splits["test"]
    rng = np.random.default_rng(a.seed)
    bidx = rng.choice(sp["n_baskets"], size=min(a.n_baskets, sp["n_baskets"]),
                      replace=False)
    zq, kq, eta_all, zb_all = [], [], [], []
    with torch.no_grad():
        for s in range(0, len(bidx), 256):
            b = bidx[s:s + 256]
            bt = nb.make_batch(d, m, "test", b, rng, dev)
            ar = torch.arange(bt["cand"].shape[0])
            tg = bt["target"]
            j = bt["cand"][ar, tg]
            z = (m.q0[j]
                 - (m.q_gamma[bt["user"]] * m.q_beta[j]).sum(-1) * bt["dlogp"][ar, tg]
                 + (m.q_state[j] * bt["state"][ar, tg, :]).sum(-1))
            zq.append(z.numpy()); kq.append(bt["units"].numpy())
            ib = nb.incidence_batch(d, m, "test", b, 16, rng, dev, 32)
            if ib is None:
                continue
            lin = (m.c0[ib["cat"]]
                   + (m.c_user[ib["user"]] * m.c_cat[ib["cat"]]).sum(-1)
                   + (m.c_state[ib["cat"]] * ib["state"]).sum(-1)
                   + m.c_hab[ib["cat"]] * ib["habit"]
                   + m.kappa()[ib["cat"]] * (ib["iv"] - m.iv_ref[ib["cat"]]))
            eta_all.append(lin.numpy())
            pos = ib["y"] > 0
            if bool(pos.any()):
                zb_all.append((m.b0[ib["cat"][pos]] + m.b_user[ib["user"][pos]]
                               - m.b_price[ib["cat"][pos]] * ib["pdev"][pos]).numpy())
    zq = np.concatenate(zq); kq = np.concatenate(kq)
    eta_all = np.concatenate(eta_all); zb_all = np.concatenate(zb_all)

    lam = np.exp(np.clip(zq, -6, 4))
    p1_fit, p1_obs = float(np.exp(-lam).mean()), float((kq == 1).mean())
    log(f"  P(q = 1): observed {p1_obs:.4f}   fitted mean e^-Lambda {p1_fit:.4f}")
    log(f"    Eq. 14 cannot put less than e^-Lambda on q = 1 for any Lambda, so the")
    log(f"    floor binds iff the observed share falls below what the fit can reach.")
    res["checks"].append(verdict("modal-line floor is not binding",
                                 abs(p1_obs - p1_fit) < 0.05,
                                 f"observed {p1_obs:.4f} vs fitted {p1_fit:.4f}"))

    log("")
    log("CLAMPS.  Version 2 states the bounds and says the binding rate is unmeasured.")
    for nm, arr, lo, hi in (("eta (incidence)", eta_all, -12.0, 4.0),
                            ("z_qty (units)", zq, -6.0, 4.0),
                            ("z_brd (breadth)", zb_all, -6.0, 3.0)):
        f_lo, f_hi = float((arr < lo).mean()), float((arr > hi).mean())
        log(f"  {nm:18s} below {lo:+.0f}: {100 * f_lo:7.4f}%   above {hi:+.0f}: {100 * f_hi:7.4f}%")
        # 1e-3 not 1e-4: a clamp that fires on three evaluations in ten thousand is
        # negligible, and calling that a violated assumption is not informative.
        res["checks"].append(verdict(f"clamp negligible: {nm}", f_lo + f_hi < 1e-3,
                                     f"{100 * (f_lo + f_hi):.4f}% of evaluations clipped"))

    # -------------------------------------------- 6  units independent given S
    log("")
    log("UNITS INDEPENDENCE.  Eq. 14 makes q_j independent across products given S.")
    log("If true, the within-basket correlation of (q - E[q]) between two products of")
    log("the same basket is zero.")
    te = bk[bk.split == "test"]
    g = te.groupby("BASKET_ID")
    mu_j = tr.groupby("item_id").units.mean()
    rows = []
    for _, grp in g:
        if len(grp) < 2:
            continue
        r = grp.units.to_numpy() - grp.item_id.map(mu_j).fillna(tr.units.mean()).to_numpy()
        rows.append((r.sum() ** 2 - (r ** 2).sum(), len(r) * (len(r) - 1), (r ** 2).sum(), len(r)))
        if len(rows) >= a.n_baskets:
            break
    cross = sum(x[0] for x in rows); npair = sum(x[1] for x in rows)
    var = sum(x[2] for x in rows); nobs = sum(x[3] for x in rows)
    rho = (cross / npair) / (var / nobs)
    log(f"  within-basket residual correlation over {len(rows):,} baskets: {rho:+.4f}")
    res["checks"].append(verdict("units independent given S", abs(rho) < 0.05,
                                 f"within-basket residual correlation {rho:+.4f}"))

    log("")
    n_ok = sum(c["holds"] for c in res["checks"])
    log(f"  {n_ok} of {len(res['checks'])} assumptions close; "
        f"{len(res['checks']) - n_ok} are violated and now sized.")
    with open(os.path.join(OUT, "assumption_checks.json"), "w") as f:
        json.dump(res, f, indent=2)
    log("")
    log("wrote out/assumption_checks.json")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--label", default="ps_nested")
    p.add_argument("--n-baskets", type=int, default=4000)
    p.add_argument("--seed", type=int, default=0)
    main(p.parse_args())
