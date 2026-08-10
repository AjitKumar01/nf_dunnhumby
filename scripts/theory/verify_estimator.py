"""
The two gaps Appendix B admitted, closed.

GAP 1.  Every stability, mode and effective-sample-size number in section 14.2 was computed
with rho_0 switched OFF, because verify_response.py builds instances without a size
potential.  Since rho_0 is the term the design turns on, those results characterised the
model being replaced rather than the model proposed.  This script recomputes the mode,
Lambda(z*), lambda_max and the importance-sampling ESS with rho_0 ACTIVE.

GAP 2.  Corollary 3 claims a sampler whose first step is sampling-importance-resampling,
but both verification scripts draw z exactly from a quadrature grid, so SIR itself was
never exercised.  It is the step that can fail.  This script runs the real thing --
Laplace proposal, importance weights, resample one -- and compares whole-basket frequencies
against enumeration, across a range of ESS.

Everything is checked against explicit enumeration of all 2^J subsets.

Run:  python3 verify_estimator.py
"""
import argparse
import itertools
import math

import numpy as np

from verify_dispersion_bound import conditional_moments, find_mode, marginal_var
from verify_normaliser import cat_of, gauss_hermite, make_instance
from verify_size_term import cat_polys, enumerate_all, integrand, sample_exactly_r


def log(m):
    print(f"[est] {m}", flush=True)


def laplace_proposal(T):
    """Mode and covariance of the Laplace approximation, with rho_0 active throughout."""
    zs = find_mode(T)
    L0, u, L, _ = conditional_moments(T, zs)
    K = T["Kz"]
    H = np.eye(K) - L                            # -Hessian of log integrand at the mode
    ev = np.linalg.eigvalsh(H)
    if ev.min() <= 1e-8:                         # not a maximum: fall back to the prior
        return zs, np.eye(K), float(np.linalg.eigvalsh(L).max()), L0
    return zs, np.linalg.inv(H), float(np.linalg.eigvalsh(L).max()), L0


def is_logz(T, n_draws, rng, zs, cov):
    """Importance-sampling estimate of log Z from the Laplace proposal, and its ESS."""
    K = T["Kz"]
    C = np.linalg.cholesky(cov)
    z = zs + rng.standard_normal((n_draws, K)) @ C.T
    lp = np.array([-0.5 * zz @ zz + math.log(max(integrand(T, zz), 1e-300)) for zz in z])
    d = z - zs
    lq = (-0.5 * np.einsum("ij,jk,ik->i", d, np.linalg.inv(cov), d)
          - 0.5 * math.log(np.linalg.det(cov)))
    lw = lp - lq
    m = lw.max()
    e = np.exp(lw - m)
    lz = m + math.log(e.mean())
    ess = float(e.sum() ** 2 / (n_draws * (e ** 2).sum()))
    return lz, ess, z, e / e.sum()


def sir_basket(T, rng, zs, cov, n_prop):
    """One draw the way Corollary 3 actually specifies it: SIR for z, then exact given z."""
    _, _, z, w = is_logz(T, n_prop, rng, zs, cov)
    zz = z[rng.choice(len(z), p=w)]
    wgt, polys = cat_polys(T, zz)
    nmax = T["J"]
    suf = [None] * (len(polys) + 1)
    suf[len(polys)] = np.array([1.0])
    for c in range(len(polys) - 1, -1, -1):
        suf[c] = np.convolve(polys[c], suf[c + 1])[:nmax + 1]
    pn = np.exp(-T["rho0"][:len(suf[0])]) * suf[0]
    n = int(rng.choice(len(pn), p=pn / pn.sum()))
    S = np.zeros(T["J"], dtype=bool)
    left = n
    for c, idx in enumerate(T["cats"]):
        tail = suf[c + 1]
        rmax = min(len(polys[c]) - 1, left)
        pr = np.array([polys[c][r] * (tail[left - r] if left - r < len(tail) else 0.0)
                       for r in range(rmax + 1)])
        r = int(rng.choice(rmax + 1, p=pr / pr.sum()))
        for pos in sample_exactly_r(wgt[idx], r, rng):
            S[idx[pos]] = True
        left -= r
    return S


def main(a):
    rng = np.random.default_rng(a.seed)
    log("=== GAP 1: mode, stability and ESS with rho_0 ACTIVE ===")
    log(f"  {'phi':>5} {'rho_0':>8} {'lam_max':>8} {'Var(n|z*)':>10} {'E[n]':>7} "
        f"{'Var(n)':>8} {'disp':>6} {'logZ err':>9} {'ESS':>6}")
    rows = []
    for scale in (0.15, 0.30, 0.45, 0.60):
        for kind in ("none", "free"):
            T = make_instance(np.random.default_rng(5), n_cat=4, per_cat=3, Kz=2,
                              phi_scale=scale, rho_scale=0.5, b_loc=-0.7)
            nn = np.arange(T["J"] + 1)
            T["rho0"] = (np.zeros(T["J"] + 1) if kind == "none"
                         else -0.45 * np.log1p(nn) + 0.02 * nn * (nn - 1) / 2.0)
            T["rho0"][0] = 0.0
            zs, cov, lam, L0 = laplace_proposal(T)
            En, Vn = marginal_var(T)
            masks, logits = enumerate_all(T)
            M = logits.max()
            true_lz = math.log(float(np.exp(logits - M).sum())) + M
            lz, ess, _, _ = is_logz(T, a.draws, rng, zs, cov)
            rows.append((scale, kind, lam, L0, En, Vn, lz - true_lz, ess, T, zs, cov,
                         masks, logits))
            log(f"  {scale:5.2f} {kind:>8} {lam:8.4f} {L0:10.4f} {En:7.3f} {Vn:8.3f} "
                f"{Vn / En:6.3f} {lz - true_lz:+9.5f} {ess:6.3f}")
    print()
    log("rho_0 raises Var(n|z*) directly -- compare the 'none' and 'free' rows at equal")
    log("phi -- which is what Proposition 3b says buys dispersion without raising lambda.")
    print()

    log("=== GAP 2: the SIR step, which nothing had exercised ===")
    log(f"  {'phi':>5} {'rho_0':>8} {'ESS':>6} {'n_prop':>7} {'TV vs truth':>12} "
        f"{'noise floor':>12} {'verdict':>8}")
    for (scale, kind, lam, L0, En, Vn, dlz, ess, T, zs, cov, masks, logits) in rows:
        if kind != "free":
            continue
        p = np.exp(logits - logits.max())
        p /= p.sum()
        key = {tuple(np.flatnonzero(m)): k for k, m in enumerate(masks)}
        for n_prop in (4, 32, 256):
            emp = np.zeros(len(masks))
            for _ in range(a.draws_sir):
                emp[key[tuple(np.flatnonzero(sir_basket(T, rng, zs, cov, n_prop)))]] += 1
            emp /= a.draws_sir
            tv = 0.5 * np.abs(emp - p).sum()
            floor = np.mean([0.5 * np.abs(rng.multinomial(a.draws_sir, p) / a.draws_sir
                                          - p).sum() for _ in range(8)])
            log(f"  {scale:5.2f} {kind:>8} {ess:6.3f} {n_prop:7d} {tv:12.5f} "
                f"{floor:12.5f} {'PASS' if tv < 2.0 * floor else 'FAIL':>8}")
    print()
    log("SIR at n_prop = 1 would be a pure proposal draw and is biased by construction;")
    log("the question is how fast the bias falls, and these rows answer it.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--draws", type=int, default=4000)
    p.add_argument("--draws-sir", type=int, default=8000)
    p.add_argument("--seed", type=int, default=3)
    main(p.parse_args())
