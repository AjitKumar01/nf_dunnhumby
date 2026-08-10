"""
Two claims about the version-3 model that decide whether it can serve as a price
environment, both checked against brute-force enumeration.

CLAIM 1 -- the size response IS the size variance.

For any exponential family with sufficient statistic x (the purchase indicator vector)
and natural parameter b, log Z is the cumulant generating function, so

    d E[n] / d delta  =  sum_{j,k} d^2 log Z / db_j db_k  =  Var(n),     n = sum_j x_j

where delta is a uniform shift added to every b_j -- exactly what a chain-wide price
change produces.  So the semi-elasticity of mean basket size to a uniform value shift
equals the DISPERSION INDEX Var(n)/E[n].  This is an identity, not an approximation.

Consequence, and the reason this script exists: a model that gets the basket-size
variance wrong gets the aggregate response to a chain-wide price move wrong by the same
factor.  Distributional fidelity and counterfactual validity are the same defect here,
not two separate desiderata.  Measured on dunnhumby training baskets, Var(n)/E[n] = 10.62.

CLAIM 2 -- dispersion is bought by approaching a phase transition.

Under the Gaussian-latent representation the mode equation is z = Phi' pi(z), whose
Jacobian is Lambda(z) = Cov(sum_j phi_j x_j | z).  If lambda_max(Lambda) < 1 the map is a
contraction, the mode is unique, and the Laplace proposal is sound.  The Laplace
expansion also gives

    Var(n)  ~=  Lambda_0 + u' (I - Lambda)^{-1} u,    Lambda_0 = sum_j pi_j (1 - pi_j),
                                                      u = sum_j pi_j (1 - pi_j) phi_j

which diverges as lambda_max -> 1.  So a single Gaussian factor can only reach a high
dispersion index by sitting near the critical point, where the estimator degrades and the
counterfactual response becomes unbounded.  This script exhibits both regimes.

Run:  python3 verify_response.py
"""
import argparse
import itertools
import math

import numpy as np

from verify_normaliser import (cat_of, enumerate_all, esp, find_mode, gauss_hermite,
                               inclusion_given_z, integrand, make_instance, stability)


def log(m):
    print(f"[rsp] {m}", flush=True)


def moments(T):
    """E[n] and Var(n) by explicit enumeration of all 2^J subsets."""
    masks, logits = enumerate_all(T)
    p = np.exp(logits - logits.max())
    p /= p.sum()
    n = masks.sum(1)
    m1 = float(p @ n)
    return m1, float(p @ (n - m1) ** 2)


def shifted(T, delta):
    U = dict(T)
    U["b"] = T["b"] + delta
    return U


def claim1(T, name, h=1e-4):
    e0, v0 = moments(T)
    ep, _ = moments(shifted(T, +h))
    em, _ = moments(shifted(T, -h))
    d = (ep - em) / (2 * h)
    log(f"{name}: E[n] {e0:.6f}  Var(n) {v0:.6f}  dispersion {v0 / e0:.4f}")
    log(f"    dE[n]/d(uniform shift)  finite difference {d:.6f}   Var(n) {v0:.6f}"
        f"   rel err {abs(d - v0) / v0:.3e}")
    return v0 / e0


def claim2(T, name, n_node=40):
    """Compare the Laplace prediction of Var(n) with the enumerated truth, and report
    where the instance sits relative to the critical point."""
    zh = find_mode(T)
    pi, _ = inclusion_given_z(T, zh)
    L = stability(T, zh)
    lam = float(np.linalg.eigvalsh(L).max())
    v = pi * (1 - pi)
    L0 = float(v.sum())
    u = T["phi"].T @ v
    pred = L0 + float(u @ np.linalg.solve(np.eye(T["Kz"]) - L, u)) if lam < 1 else np.nan
    _, v_true = moments(T)
    log(f"{name}: lambda_max {lam:.4f}   Var(n) enumerated {v_true:.4f}"
        f"   Laplace prediction {pred:.4f}")
    return lam, v_true, pred


def multi_start_fixed_points(T, n_start=12, seed=0):
    """How many distinct solutions does z = Phi' pi(z) have?  One if the map contracts."""
    rng = np.random.default_rng(seed)
    sols = []
    for _ in range(n_start):
        z = rng.normal(0, 4.0, T["Kz"])
        for _ in range(250):
            z = T["phi"].T @ inclusion_given_z(T, z)[0]
        if not any(np.linalg.norm(z - s) < 1e-4 for s in sols):
            sols.append(z.copy())
    return sols


def laplace_ess(T, n_draws=4000, seed=0):
    """Effective sample size of importance sampling from a Laplace proposal at the mode
    -- the diagnostic that decides whether the estimator is usable."""
    rng = np.random.default_rng(seed)
    zh = find_mode(T)
    L = stability(T, zh)
    H = np.eye(T["Kz"]) - L                       # -Hessian of log integrand at the mode
    ev = np.linalg.eigvalsh(H)
    if ev.min() <= 1e-6:                          # not a maximum: proposal is undefined
        cov = np.eye(T["Kz"])
    else:
        cov = np.linalg.inv(H)
    Lc = np.linalg.cholesky(cov)
    z = zh + rng.standard_normal((n_draws, T["Kz"])) @ Lc.T
    lp = np.array([-0.5 * zz @ zz + math.log(max(integrand(T, zz, T["per_cat"]), 1e-300))
                   for zz in z])
    d = z - zh
    lq = -0.5 * np.einsum("ij,jk,ik->i", d, np.linalg.inv(cov), d) \
        - 0.5 * math.log(np.linalg.det(cov))
    w = lp - lq
    w -= w.max()
    e = np.exp(w)
    return float(e.sum() ** 2 / (len(e) * (e ** 2).sum()))


def main(a):
    rng = np.random.default_rng(a.seed)
    log("=== CLAIM 1: dE[n]/d(uniform shift) = Var(n) ===")
    for nm, cfg in [("weak  ", dict(n_cat=4, per_cat=3, Kz=2, phi_scale=0.25,
                                    rho_scale=0.4, b_loc=-1.0)),
                    ("strong", dict(n_cat=4, per_cat=3, Kz=2, phi_scale=0.55,
                                    rho_scale=1.2, b_loc=-0.5)),
                    ("dense ", dict(n_cat=3, per_cat=4, Kz=2, phi_scale=0.40,
                                    rho_scale=0.8, b_loc=+0.3))]:
        claim1(make_instance(rng, **cfg), nm)
    print()

    log("=== CLAIM 2: dispersion, stability, and the estimator ===")
    log("one shared factor, phi_j = a for every product.  a rises toward the point where")
    log("the mode map stops contracting.  16 products, so 'buy everything' is n = 16 and")
    log("the high phase is visible as saturation rather than as a true divergence.")
    n_cat, per = 4, 4
    J = n_cat * per
    base = make_instance(np.random.default_rng(7), n_cat=n_cat, per_cat=per, Kz=1,
                         phi_scale=0.0, rho_scale=0.0, b_loc=-2.6)
    base["rho"] = np.zeros(n_cat)
    rows = []
    for aa in (0.10, 0.25, 0.40, 0.50, 0.55, 0.60, 0.65, 0.75, 0.90):
        T = dict(base)
        T["phi"] = np.full((J, 1), aa)
        lam, vt, pred = claim2(T, f"  a={aa:.2f}")
        rows.append((aa, lam, moments(T)[0], vt, pred,
                     len(multi_start_fixed_points(T)), laplace_ess(T)))
    print()
    log(f"  {'a':>5} {'lambda':>8} {'E[n]':>7} {'Var(n)':>9} {'disp':>7} "
        f"{'Laplace':>9} {'err':>8} {'modes':>6} {'ESS':>7}")
    for aa, lam, e0, vt, pred, nm, ess in rows:
        err = abs(pred - vt) / vt if np.isfinite(pred) else np.nan
        log(f"  {aa:5.2f} {lam:8.4f} {e0:7.3f} {vt:9.3f} {vt / e0:7.3f} "
            f"{pred:9.3f} {err:7.1%} {nm:6d} {ess:7.3f}")
    print()
    log("dunnhumby training baskets: E[n] 7.803, Var(n) 82.842, dispersion 10.617.")
    log("The Laplace variance formula solved for lambda at that dispersion:")
    E, V = 7.803, 82.842
    L0 = E * (1 - E / 2800.0)                  # sum pi(1-pi), assortment about 2,800
    # one factor, phi_j = a: u = a L0, Lambda = a^2 L0, so V = L0 + a^2 L0^2/(1 - a^2 L0)
    x = (V - L0) / (L0 ** 2 + (V - L0) * L0)   # x = a^2
    log(f"  Lambda_0 = {L0:.3f},  a^2 = {x:.4f},  lambda = a^2 Lambda_0 = {x * L0:.4f}")
    log("  and the table shows this formula OVERSTATES Var(n) near the critical point,")
    log("  so the lambda actually required is higher still.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    main(p.parse_args())
