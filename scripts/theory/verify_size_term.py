"""
The size term, and why the design needs it.

verify_response.py establishes, and this script's companion arithmetic confirms, a
tight relation between two things a retail environment must get right at once.

Under the Gaussian-latent representation the Laplace expansion gives

    Var(n)  =  Lambda_0 / (1 - lambda_size),     Lambda_0 = sum_j pi_j (1 - pi_j)

where lambda_size is the eigenvalue of the stability matrix along the direction every
product loads on with the same sign.  Since Lambda_0 ~= E[n] when purchase probabilities
are small, the dispersion index is

    Var(n) / E[n]  ~=  1 / (1 - lambda_size),    i.e.   lambda_size = 1 - 1/dispersion.

Combined with  dE[n]/d(uniform shift) = Var(n)  this says the aggregate basket-size
response to a chain-wide price move is exactly the quantity that diverges at the
critical point.  Reproducing dunnhumby's dispersion from a shared Gaussian factor means
running the model at lambda = 0.77 (after the covariates the model already has) or 0.91
(raw) -- and the enumerable toy systems show importance sampling degrading well before
that.  The size direction therefore must NOT be carried by the Gaussian latent.

The fix is to carry it by explicit enumeration, using the same device already used for
the within-category count.  Add a free weight on total basket size:

    E(S) = sum_{j in S} b_j + sum_{{j,k} subset S} phi_j'phi_k
           - sum_c rho_c C(n_c, 2)  -  rho_0(n),     n = |S|

Then, with A_n(z) the coefficients of the truncated polynomial product

    prod_c ( sum_r exp(-rho_c C(r,2)) e_r(w^(c)(z)) u^r )  =  sum_n A_n(z) u^n,

    Z = E_z [ sum_n exp(-rho_0(n)) A_n(z) ].

rho_0 is a free function of n, so the size law is a free discrete distribution rather
than a consequence of the interaction -- and its response to a price move is whatever
the data says, not whatever the critical point allows.

This script checks that identity, and the exact sampler that goes with it, against
enumeration of all 2^J subsets.

Run:  python3 verify_size_term.py
"""
import argparse
import itertools
import math

import numpy as np

from verify_normaliser import cat_of, esp, gauss_hermite, make_instance


def log(m):
    print(f"[siz] {m}", flush=True)


# --------------------------------------------------------------------- brute force


def enumerate_all(T):
    J, phi, b, rho, rho0, co = (T["J"], T["phi"], T["b"], T["rho"], T["rho0"], cat_of(T))
    masks, logits = [], []
    for bits in itertools.product([0, 1], repeat=J):
        m = np.array(bits, dtype=bool)
        n = int(m.sum())
        if n:
            v = phi[m].sum(0)
            pair = 0.5 * (v @ v - (phi[m] ** 2).sum())
            n_c = np.bincount(co[m], minlength=T["n_cat"])
            E = float(b[m].sum()) + pair - float(rho @ (n_c * (n_c - 1) / 2.0))
        else:
            E = 0.0
        masks.append(m)
        logits.append(E - rho0[n])
    return np.array(masks), np.array(logits)


# ------------------------------------------------------------------- the formula


def cat_polys(T, z):
    """Per-category generating polynomial, coefficient r = exp(-rho_c C(r,2)) e_r(w)."""
    w = np.exp(T["b"] - 0.5 * (T["phi"] ** 2).sum(1) + T["phi"] @ z)
    out = []
    for c, idx in enumerate(T["cats"]):
        N = len(idx)
        r = np.arange(N + 1)
        out.append(np.exp(-T["rho"][c] * r * (r - 1) / 2.0) * esp(w[idx], N))
    return w, out


def size_coeffs(polys, nmax):
    """A_n: convolve the per-category polynomials, truncating at degree nmax."""
    A = np.zeros(nmax + 1)
    A[0] = 1.0
    for p in polys:
        A = np.convolve(A, p)[:nmax + 1]
    return A


def integrand(T, z):
    _, polys = cat_polys(T, z)
    A = size_coeffs(polys, T["J"])
    return float(np.exp(-T["rho0"]) @ A)


def Z_quadrature(T, n_node):
    gz, gw = gauss_hermite(T["Kz"], n_node)
    return float(sum(gw[i] * integrand(T, gz[i]) for i in range(len(gz))))


# ----------------------------------------------------------------------- sampling


def sample_exactly_r(w, r, rng):
    N = len(w)
    if r == 0:
        return []
    suf = np.zeros((N + 1, r + 1))
    suf[N, 0] = 1.0
    for i in range(N - 1, -1, -1):
        suf[i] = suf[i + 1]
        suf[i, 1:] = suf[i + 1, 1:] + w[i] * suf[i + 1, :-1]
    out, need = [], r
    for i in range(N):
        if need == 0:
            break
        if N - i == need:
            out.extend(range(i, N))
            break
        p = w[i] * suf[i + 1, need - 1] / suf[i, need]
        if rng.random() < p:
            out.append(i)
            need -= 1
    return out


def sample_basket(T, rng, gz, post):
    """Exact draw.  Three nested steps, each exact:
       1  z from its posterior;
       2  the total size n, then the per-category counts given n, by a backward pass
          over the same polynomial product used for Z;
       3  the items within each category, by the suffix-ESP sequential algorithm."""
    z = gz[rng.choice(len(gz), p=post)]
    w, polys = cat_polys(T, z)
    nmax = T["J"]
    # suffix products: suf[c] = polynomial of categories c..C-1
    suf = [None] * (len(polys) + 1)
    suf[len(polys)] = np.array([1.0])
    for c in range(len(polys) - 1, -1, -1):
        suf[c] = np.convolve(polys[c], suf[c + 1])[:nmax + 1]
    A = suf[0]
    pn = np.exp(-T["rho0"][:len(A)]) * A
    n = int(rng.choice(len(pn), p=pn / pn.sum()))
    S = np.zeros(T["J"], dtype=bool)
    left = n
    for c, idx in enumerate(T["cats"]):
        tail = suf[c + 1]
        rmax = min(len(polys[c]) - 1, left)
        pr = np.array([polys[c][r] * (tail[left - r] if left - r < len(tail) else 0.0)
                       for r in range(rmax + 1)])
        r = int(rng.choice(rmax + 1, p=pr / pr.sum()))
        for pos in sample_exactly_r(w[idx], r, rng):
            S[idx[pos]] = True
        left -= r
    return S


# --------------------------------------------------------------------------- main


def run_case(name, T, n_node, rng, n_draws):
    masks, logits = enumerate_all(T)
    M = logits.max()
    Z_bf = float(np.exp(logits - M).sum()) * math.exp(M)
    Z_q = Z_quadrature(T, n_node)
    log(f"{name}: J={T['J']} ({T['n_cat']}x{T['per_cat']}), Kz={T['Kz']}, "
        f"2^J={2 ** T['J']:,} subsets")
    log(f"    log Z  brute force {math.log(Z_bf):+.10f}   formula {math.log(Z_q):+.10f}"
        f"   rel err {abs(Z_q - Z_bf) / Z_bf:.3e}")

    p = np.exp(logits - M)
    p /= p.sum()
    n = masks.sum(1)
    # the size law, which is the object rho_0 is there to control
    law_bf = np.array([p[n == k].sum() for k in range(T["J"] + 1)])
    gz, gw = gauss_hermite(T["Kz"], n_node)
    integ = np.array([integrand(T, z) for z in gz])
    post = gw * integ
    post /= post.sum()
    law_f = np.zeros(T["J"] + 1)
    for i, z in enumerate(gz):
        A = size_coeffs(cat_polys(T, z)[1], T["J"])
        q = np.exp(-T["rho0"]) * A
        law_f += post[i] * q / q.sum()
    log(f"    size law P(n=k): max abs err {np.abs(law_f - law_bf).max():.3e}"
        f"   mean size {float(law_bf @ np.arange(T['J'] + 1)):.4f}")

    key = {tuple(np.flatnonzero(m)): k for k, m in enumerate(masks)}
    emp = np.zeros(len(masks))
    for _ in range(n_draws):
        emp[key[tuple(np.flatnonzero(sample_basket(T, rng, gz, post)))]] += 1
    emp /= n_draws
    tv = 0.5 * np.abs(emp - p).sum()
    floor = np.mean([0.5 * np.abs(rng.multinomial(n_draws, p) / n_draws - p).sum()
                     for _ in range(12)])
    log(f"    sampler: total variation vs enumerated law {tv:.5f}"
        f"   noise floor {floor:.5f}   {'PASS' if tv < 2.0 * floor else 'FAIL'}")
    return abs(Z_q - Z_bf) / Z_bf


def main(a):
    rng = np.random.default_rng(a.seed)
    worst = 0.0
    cases = [
        ("convex size penalty ", dict(n_cat=4, per_cat=3, Kz=2, phi_scale=0.30,
                                      rho_scale=0.5, b_loc=-0.6),
         lambda n: 0.06 * n * (n - 1) / 2),
        ("concave, size reward", dict(n_cat=4, per_cat=3, Kz=2, phi_scale=0.30,
                                      rho_scale=0.5, b_loc=-1.2),
         lambda n: -0.9 * np.log1p(n)),
        ("free-form weight    ", dict(n_cat=3, per_cat=4, Kz=2, phi_scale=0.35,
                                      rho_scale=0.7, b_loc=-0.4),
         None),
        ("no size term (rho0=0)", dict(n_cat=4, per_cat=3, Kz=2, phi_scale=0.30,
                                       rho_scale=0.5, b_loc=-0.6),
         lambda n: 0.0 * n),
    ]
    for name, cfg, fn in cases:
        T = make_instance(rng, **cfg)
        n = np.arange(T["J"] + 1)
        T["rho0"] = (rng.normal(0, 1.0, T["J"] + 1) if fn is None
                     else np.asarray(fn(n), dtype=float) * np.ones(T["J"] + 1))
        T["rho0"][0] = 0.0                       # the empty basket anchors the scale
        worst = max(worst, run_case(name, T, a.nodes, rng, a.draws))
        print()
    log(f"worst relative error in Z across {len(cases)} cases: {worst:.3e}")
    print()
    log("the dispersion-stability identity, on dunnhumby numbers:")
    for lab, D in (("raw", 10.617), ("after household x day-of-week x gap bin", 4.352)):
        log(f"    dispersion {D:6.3f}  ->  a Gaussian factor would need "
            f"lambda_size = 1 - 1/D = {1 - 1 / D:.4f}   ({lab})")
    log("    with the size term, lambda_size is free to be small and rho_0 carries it.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--nodes", type=int, default=32)
    p.add_argument("--draws", type=int, default=40000)
    p.add_argument("--seed", type=int, default=1)
    main(p.parse_args())
