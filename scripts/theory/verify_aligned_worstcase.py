"""
The aligned worst case, with rho_0 switched ON.

Section 14.2's bifurcation sweep uses the worst case for stability: every product carries
the SAME loading phi_j = a, so all the co-purchase feedback points one way.  That sweep runs
at rho_0 = 0, and section 14.4 claims the size potential is what keeps the model out of the
critical regime.  The claim was never tested in the worst case: verify_estimator.py only
reached lambda_max 0.556 because it used random loadings.

This closes it.  Same aligned family, 16 products, sweeping the loading a with

    rho_0(n) = rho * n(n-1)/2,   rho in {0, 0.02, 0.05, 0.10}

reporting, against full enumeration of all 2^16 subsets: lambda_max at the located mode, the
NUMBER OF DISTINCT MODES by multi-start, E[n], Var(n), dispersion, the importance-sampling
ESS, and the error in log Z.

Prediction under test: a convex rho_0 raises the marginal cost of each extra line, opposing
the runaway that drives the bifurcation, so it should hold one mode and a usable ESS at
values of a where rho_0 = 0 already has two.

IMPLEMENTATION NOTE.  Everything is precomputed once per instance as flat arrays over the
65,536 subsets, so a conditional moment is a matrix-vector product rather than a Python loop.
A first version of this script called a loop-based enumerator inside the mode iteration and
would have taken tens of hours; this one takes seconds.  The mathematics is identical.

Run:  python3 verify_aligned_worstcase.py
"""
import argparse
import itertools
import math

import numpy as np


def log(m):
    print(f"[awc] {m}", flush=True)


class Enum:
    """All 2^J subsets, with everything that does not depend on z precomputed."""

    def __init__(self, J, cats):
        self.J, self.cats = J, cats
        self.masks = np.array(list(itertools.product([0, 1], repeat=J)), dtype=np.int8)
        self.n = self.masks.sum(1)
        self.nc = np.stack([self.masks[:, idx].sum(1) for idx in cats], axis=1)
        self.pairs_c = (self.nc * (self.nc - 1) // 2).astype(float)

    def prepare(self, b, phi, rho, rho0):
        bt = b - 0.5 * (phi ** 2).sum(1)
        self.base = (self.masks @ bt) - self.pairs_c @ rho - rho0[self.n]
        self.proj = self.masks @ phi                       # v_S for every subset
        self.half = 0.5 * (self.proj ** 2).sum(1)
        return self

    def true_logz(self):
        return float(logsumexp(self.base + self.half))

    def log_f(self, z):
        """log of the Theorem-1 integrand at z, computed exactly from the enumeration."""
        return float(logsumexp(self.base + self.proj @ z))

    def moments(self, z):
        """E[n|z], Var(n|z), Cov(n, v_S|z), Cov(v_S|z), E[v_S|z] -- all exact."""
        lg = self.base + self.proj @ z
        p = np.exp(lg - lg.max())
        p /= p.sum()
        En = float(p @ self.n)
        Ev = p @ self.proj
        dn = self.n - En
        dv = self.proj - Ev
        return (En, float(p @ dn ** 2), (p * dn) @ dv, (dv * p[:, None]).T @ dv, Ev)

    def marginal(self, gz, gw):
        """Exact E[n], Var(n) by integrating the enumeration over the quadrature grid."""
        tot = np.zeros(len(self.base))
        for i in range(len(gz)):
            lg = self.base + self.proj @ gz[i]
            tot += gw[i] * np.exp(lg - lg.max()) * math.exp(lg.max())
        p = tot / tot.sum()
        En = float(p @ self.n)
        return En, float(p @ (self.n - En) ** 2)


def logsumexp(x):
    m = x.max()
    return m + np.log(np.exp(x - m).sum())


def modes(E, Kz, n_start=8, iters=300, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for k in range(n_start):
        z = np.zeros(Kz) if k == 0 else rng.normal(0, 5.0, Kz)
        for _ in range(iters):
            z = E.moments(z)[4]
        if not any(np.linalg.norm(z - m) < 1e-4 for m in out):
            out.append(z.copy())
    return out


def ess_logz(E, zs, cov, n_draws, rng):
    Kz = len(zs)
    C = np.linalg.cholesky(cov)
    z = zs + rng.standard_normal((n_draws, Kz)) @ C.T
    lp = np.array([-0.5 * zz @ zz + E.log_f(zz) for zz in z])
    d = z - zs
    lq = (-0.5 * np.einsum("ij,jk,ik->i", d, np.linalg.inv(cov), d)
          - 0.5 * math.log(np.linalg.det(cov)))
    lw = lp - lq
    m = lw.max()
    e = np.exp(lw - m)
    return (m + math.log(e.mean()),
            float(e.sum() ** 2 / (n_draws * (e ** 2).sum())))


def main(a):
    rng = np.random.default_rng(a.seed)
    n_cat, per = 4, 4
    J = n_cat * per
    cats = [np.arange(c * per, (c + 1) * per) for c in range(n_cat)]
    E = Enum(J, cats)
    g = np.random.default_rng(7)
    b = g.normal(-2.6, 0.8, J)
    rho_c = np.zeros(n_cat)
    nn = np.arange(J + 1)
    x, w = np.polynomial.hermite.hermgauss(48)
    gz, gw = (np.sqrt(2.0) * x).reshape(-1, 1), w / math.sqrt(math.pi)

    log(f"aligned loadings phi_j = a for all {J} products; rho_c = 0; "
        f"rho_0(n) = rho n(n-1)/2; {2**J:,} subsets enumerated")
    log("")
    log(f"  {'rho':>5} {'a':>5} {'lam_max':>8} {'modes':>6} {'E[n]':>7} {'Var(n)':>9} "
        f"{'disp':>6} {'ESS':>6} {'logZ err':>9}")
    first = {}
    for rho in (0.0, 0.02, 0.05, 0.10):
        rho0 = rho * nn * (nn - 1) / 2.0
        rho0[0] = 0.0
        split = None
        for aa in (0.30, 0.45, 0.55, 0.60, 0.65, 0.75, 0.90):
            phi = np.full((J, 1), aa)
            E.prepare(b, phi, rho_c, rho0)
            ms = modes(E, 1)
            zs = ms[0]
            _, L0, u, L, _ = E.moments(zs)
            lam = float(np.linalg.eigvalsh(np.atleast_2d(L)).max())
            En, Vn = E.marginal(gz, gw)
            H = np.eye(1) - np.atleast_2d(L)
            cov = np.linalg.inv(H) if float(H[0, 0]) > 1e-8 else np.eye(1)
            lz, ess = ess_logz(E, zs, cov, a.draws, rng)
            err = lz - E.true_logz()
            if len(ms) > 1 and split is None:
                split = aa
            log(f"  {rho:5.2f} {aa:5.2f} {lam:8.4f} {len(ms):6d} {En:7.3f} {Vn:9.3f} "
                f"{Vn / max(En,1e-9):6.3f} {ess:6.3f} {err:+9.4f}")
        first[rho] = split
        log("")

    log("first loading a at which a second mode appears:")
    for rho, sp in first.items():
        log(f"    rho_0 slope {rho:5.2f} ->  "
            f"{'none in this sweep' if sp is None else f'a = {sp:.2f}'}")
    log("")
    log("Read the ESS column, not lambda_max: after a bifurcation each mode is locally")
    log("stable with lambda_max < 1, so the eigenvalue at the located mode is misleading.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--draws", type=int, default=3000)
    p.add_argument("--seed", type=int, default=3)
    main(p.parse_args())
