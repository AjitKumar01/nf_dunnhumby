"""
The dispersion-stability bound -- corrected, and with the error that preceded it recorded.

WHAT WAS WRONG.  An earlier draft wrote the bound as

    Var(n) <= Lambda_0 / (1 - lambda_max),      Lambda_0 = sum_j pi_j (1 - pi_j)      (WRONG)

and, using Lambda_0 ~= E[n], concluded that reproducing a dispersion index D forces
lambda_max >= 1 - 1/D: 0.906 on dunnhumby's raw basket sizes.

sum_j pi_j(1-pi_j) is Var(n | z) ONLY when the purchase indicators are conditionally
independent given z.  The within-category term rho_c and the size potential rho_0 are
exactly what destroys that independence -- and they are the two terms the design turns on.
The counterexample is flat: take phi = 0, so lambda_max = 0 exactly, and choose rho_0 so
that P(n) puts half its mass at 0 and half at 12 over 12 products.  Then Var(n) = 36 while
sum pi(1-pi) = 3, and the "bound" is violated twelve-fold.  Worse, the wrong form
contradicted the design it was written to justify: section 14.4 says rho_0 frees the model
to have large Var(n) at small lambda, which the wrong bound forbids.

THE CORRECT STATEMENT.  Lambda_0 is the CONDITIONAL variance and u the CONDITIONAL
covariance, both at the mode:

    Lambda_0 := Var(n | z*),   u := Cov(n, v_S | z*),   Lambda := Cov(v_S | z*),
    v_S := sum_{j in S} phi_j

    Var(n)  =  Lambda_0 + u' (I - Lambda)^{-1} u     (Laplace order)
            <= Lambda_0 / (1 - lambda_max(Lambda))                                     (*)

PROOF of (*).  With e = u/||u||, u'(I-Lambda)^{-1}u = ||u||^2 e'(I-Lambda)^{-1}e <=
||u||^2 / (1 - lambda_max).  By Cauchy-Schwarz on the conditional covariance,
||u||^2 = Cov(n, e'v_S)^2 <= Var(n|z*) Var(e'v_S|z*) = Lambda_0 e'Lambda e <=
Lambda_0 lambda_max.  Hence u'(I-Lambda)^{-1}u <= Lambda_0 lambda_max/(1-lambda_max), and
adding Lambda_0 gives (*).  QED

WHAT SURVIVES OF THE ORIGINAL ARGUMENT, and it is the part the design needs.  In the
sub-model with rho_c = rho_0 = 0 the indicators ARE conditionally independent, so
Lambda_0 = sum pi(1-pi) ~= E[n] and (*) collapses to lambda_max >= 1 - 1/D.  That
sub-model is precisely a basket model whose only interaction is a low-rank Gaussian
latent -- and it is the one that cannot reach dunnhumby's dispersion without going
critical.  Switching rho_0 on raises Lambda_0 directly and lets lambda_max stay small.
So the corrected bound says what the design claims, instead of contradicting it.

Run:  python3 verify_dispersion_bound.py
"""
import argparse
import itertools
from math import comb

import numpy as np

from verify_normaliser import cat_of, make_instance


def log(m):
    print(f"[bnd] {m}", flush=True)


def conditional_moments(T, z):
    """Var(n | z), Cov(n, v_S | z), Cov(v_S | z) and E[v_S | z], exactly, by enumerating
    the subsets and weighting each by its z-conditional probability."""
    J, phi, b, rho, rho0, co = (T["J"], T["phi"], T["b"], T["rho"], T["rho0"], cat_of(T))
    bt = b - 0.5 * (phi ** 2).sum(1) + phi @ z
    ns, vs, lg = [], [], []
    for bits in itertools.product([0, 1], repeat=J):
        m = np.array(bits, dtype=bool)
        n = int(m.sum())
        n_c = np.bincount(co[m], minlength=T["n_cat"])
        lg.append(float(bt[m].sum()) - float(rho @ (n_c * (n_c - 1) / 2.0)) - rho0[n])
        ns.append(n)
        vs.append(phi[m].sum(0))
    lg = np.array(lg)
    p = np.exp(lg - lg.max())
    p /= p.sum()
    ns = np.array(ns, dtype=float)
    vs = np.array(vs)
    En = float(p @ ns)
    Ev = p @ vs
    dn, dv = ns - En, vs - Ev
    return (float(p @ dn ** 2), (p * dn) @ dv, (dv * p[:, None]).T @ dv, Ev)


def find_mode(T, iters=300):
    z = np.zeros(T["Kz"])
    for _ in range(iters):
        z = conditional_moments(T, z)[3]
    return z


def marginal_var(T, n_node=32):
    """Exact E[n], Var(n) by enumerating subsets and integrating z on a quadrature grid."""
    from verify_normaliser import gauss_hermite
    gz, gw = gauss_hermite(T["Kz"], n_node)
    J, phi, b, rho, rho0, co = (T["J"], T["phi"], T["b"], T["rho"], T["rho0"], cat_of(T))
    masks = np.array([np.array(x, dtype=bool)
                      for x in itertools.product([0, 1], repeat=J)])
    n = masks.sum(1)
    base = np.zeros(len(masks))
    for k, m in enumerate(masks):
        n_c = np.bincount(co[m], minlength=T["n_cat"])
        base[k] = (float((b - 0.5 * (phi ** 2).sum(1))[m].sum())
                   - float(rho @ (n_c * (n_c - 1) / 2.0)) - rho0[int(n[k])])
    proj = masks.astype(float) @ phi                       # v_S for every subset
    tot = np.zeros(len(masks))
    for i, z in enumerate(gz):
        lg = base + proj @ z
        tot += gw[i] * np.exp(lg)
    p = tot / tot.sum()
    En = float(p @ n)
    return En, float(p @ (n - En) ** 2)


def counterexample():
    log("=== the counterexample that refutes the earlier form ===")
    J = 12
    for name, pn in [("half at 0, half at 12",
                      {**{k: 0.0 for k in range(13)}, 0: .5, 12: .5}),
                     ("uniform on 0..12", {k: 1 / 13 for k in range(13)})]:
        masks = np.array([np.array(x, dtype=bool)
                          for x in itertools.product([0, 1], repeat=J)])
        n = masks.sum(1)
        p = np.array([pn[k] / comb(J, k) for k in n])
        p /= p.sum()
        En = float(p @ n)
        Vn = float(p @ (n - En) ** 2)
        pi = (p[:, None] * masks).sum(0)
        old = float((pi * (1 - pi)).sum())
        log(f"  {name:22s} phi = 0 so lambda_max = 0 exactly;  Var(n) {Vn:6.3f}")
        log(f"  {'':22s}   old Lambda_0 = sum pi(1-pi) = {old:6.3f} -> bound {old:6.3f}"
            f"  VIOLATED {Vn / old:5.2f}x")
        log(f"  {'':22s}   new Lambda_0 = Var(n|z)     = {Vn:6.3f} -> bound {Vn:6.3f}"
            f"  holds with equality")
    print()


def main(a):
    counterexample()
    rng = np.random.default_rng(a.seed)
    log("=== the corrected bound, with rho_c and rho_0 ACTIVE ===")
    log(f"  {'Kz':>3} {'rho_0':>10} {'lam_max':>8} {'Lam_0':>7} {'Laplace':>9} "
        f"{'bound':>8} {'exact Var':>10} {'alg':>4} {'exact':>6}")
    bad = alg_bad = 0
    for Kz in (1, 2, 3):
        for kind in ("none", "convex", "free"):
            for _ in range(a.trials):
                T = make_instance(rng, n_cat=4, per_cat=3, Kz=Kz,
                                  phi_scale=float(rng.uniform(0.15, 0.5)),
                                  rho_scale=float(rng.uniform(0.0, 1.0)),
                                  b_loc=float(rng.uniform(-1.4, -0.2)))
                nn = np.arange(T["J"] + 1)
                T["rho0"] = {"none": np.zeros(T["J"] + 1),
                             "convex": 0.05 * nn * (nn - 1) / 2.0,
                             "free": rng.normal(0, 0.8, T["J"] + 1)}[kind]
                T["rho0"][0] = 0.0
                zs = find_mode(T)
                L0, u, L, _ = conditional_moments(T, zs)
                lam = float(np.linalg.eigvalsh(L).max())
                if lam >= 0.995:
                    continue
                lap = L0 + float(u @ np.linalg.solve(np.eye(Kz) - L, u))
                bound = L0 / (1 - lam)
                _, ev = marginal_var(T)
                ok, ok2 = lap <= bound * (1 + 1e-9), ev <= bound * (1 + 1e-9)
                alg_bad += (not ok)
                bad += (not ok2)
                log(f"  {Kz:3d} {kind:>10} {lam:8.4f} {L0:7.3f} {lap:9.3f} {bound:8.3f} "
                    f"{ev:10.3f} {'yes' if ok else 'NO':>4} {'yes' if ok2 else 'NO':>6}")
    print()
    log(f"algebraic bound violated: {alg_bad} (expect 0)")
    log(f"bound on the EXACT marginal variance violated: {bad}")
    log("the second column can fail: the bound is on the Laplace-order expression, and")
    log("the Laplace step itself has error -- measured and signed in verify_response.py.")
    print()
    log("=== what this leaves of the dunnhumby argument ===")
    log("Valid ONLY for the sub-model with rho_c = rho_0 = 0, where the indicators are")
    log("conditionally independent and Lambda_0 = sum pi(1-pi) ~= E[n]:")
    for lab, E, V in (("raw", 7.803, 82.842), ("residual after covariates", 7.803, 33.957)):
        L0 = E * (1 - E / 3041.0)
        log(f"    dispersion {V / E:6.3f} -> lambda_max >= {1 - L0 / V:.4f}   ({lab})")
    log("With rho_0 free, Lambda_0 = Var(n|z*) is itself free, so lambda_max need not be")
    log("large.  That is the whole point of the size term, and the corrected bound now")
    log("says so instead of forbidding it.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--trials", type=int, default=1)
    p.add_argument("--seed", type=int, default=11)
    main(p.parse_args())
