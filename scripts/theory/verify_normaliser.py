"""
Brute-force verification of the version-3 normaliser identity.

The model is a pairwise exponential family over SUBSETS of the whole store assortment --
no composition stage, no conditioning on which categories were entered:

    E(S) = sum_{j in S} b_j  +  sum_{{j,k} subset S} W_jk,
    W_jk = phi_j' phi_k  -  rho_{c(j)} 1{c(j) = c(k)},
    P(S) = exp E(S) / Z,      Z = sum over ALL 2^J subsets.

Claim (Theorem 1 of the doc).  Writing  b~_j = b_j - ||phi_j||^2 / 2  and
w_j(z) = exp(b~_j + z'phi_j),

    Z = E_{z ~ N(0, I_Kz)} [ prod_c  Psi_c(z) ],
    Psi_c(z) = sum_{r=0}^{N_c} exp(-rho_c r(r-1)/2) e_r( w^{(c)}(z) )

with e_r the elementary symmetric polynomial.  A sum over 2^J subsets becomes a
K_z-dimensional Gaussian integral of a product of C one-dimensional polynomials.

Everything here is checked against explicit enumeration, so nothing is taken on trust:

  1  Z itself.
  2  P(S = empty) = 1/Z -- exact, because the integrand at S = empty is 1 for every z.
  3  inclusion probabilities P(j in S) from d log Z / d b_j.
  4  the distribution of the within-category count.
  5  the mode equation z = Phi' pi(z) and the stability matrix Lambda(z).
  6  the exact sampler, against enumerated subset probabilities.

Run:  python3 verify_normaliser.py
"""
import argparse
import itertools
import math

import numpy as np


def log(m):
    print(f"[ver] {m}", flush=True)


# --------------------------------------------------------------------------- pieces


def esp(w, kmax):
    """e_0..e_kmax of the weights w, by the O(N k) recursion."""
    e = np.zeros(kmax + 1)
    e[0] = 1.0
    for wi in w:
        e[1:] = e[1:] + wi * e[:-1]
    return e


def psi(w, rho, kmax):
    """Psi_c = sum_r exp(-rho r(r-1)/2) e_r(w)."""
    r = np.arange(kmax + 1)
    return float(np.exp(-rho * r * (r - 1) / 2.0) @ esp(w, kmax))


def make_instance(rng, n_cat, per_cat, Kz, phi_scale, rho_scale, b_loc):
    J = n_cat * per_cat
    cats = [np.arange(c * per_cat, (c + 1) * per_cat) for c in range(n_cat)]
    return dict(
        cats=cats, J=J, n_cat=n_cat, per_cat=per_cat, Kz=Kz,
        b=rng.normal(b_loc, 0.8, J),
        phi=rng.normal(0.0, phi_scale, (J, Kz)),
        # both signs: rho > 0 is within-category repulsion, rho < 0 attraction
        rho=rng.normal(0.0, rho_scale, n_cat),
    )


def cat_of(T):
    c = np.empty(T["J"], dtype=int)
    for k, idx in enumerate(T["cats"]):
        c[idx] = k
    return c


# ------------------------------------------------------------------- brute force


def enumerate_all(T):
    """Every subset of the assortment, with its unnormalised weight.  2^J of them."""
    J, phi, b, rho, co = T["J"], T["phi"], T["b"], T["rho"], cat_of(T)
    masks, logits = [], []
    for bits in itertools.product([0, 1], repeat=J):
        m = np.array(bits, dtype=bool)
        if m.any():
            v = phi[m].sum(0)
            pair = 0.5 * (v @ v - (phi[m] ** 2).sum())        # sum_{j<k} phi_j'phi_k
            n_c = np.bincount(co[m], minlength=T["n_cat"])
            pen = float(rho @ (n_c * (n_c - 1) / 2.0))
            E = float(b[m].sum()) + pair - pen
        else:
            E = 0.0
        masks.append(m)
        logits.append(E)
    return np.array(masks), np.array(logits)


# --------------------------------------------------------------------- the formula


def gauss_hermite(Kz, n_node):
    """Nodes and weights for E_{z~N(0,I)}[f(z)] by tensor-product Gauss-Hermite."""
    x, w = np.polynomial.hermite.hermgauss(n_node)
    z = np.sqrt(2.0) * x
    grid_z = np.array(list(itertools.product(*[z] * Kz)))
    grid_w = np.prod(np.array(list(itertools.product(*[w] * Kz))), axis=1)
    return grid_z, grid_w / (math.pi ** (Kz / 2.0))


def integrand(T, z, kmax):
    """prod_c Psi_c(z), the quantity whose Gaussian mean is Z."""
    b_t = T["b"] - 0.5 * (T["phi"] ** 2).sum(1)
    w = np.exp(b_t + T["phi"] @ z)
    out = 1.0
    for c, idx in enumerate(T["cats"]):
        out *= psi(w[idx], T["rho"][c], min(kmax, len(idx)))
    return out


def Z_quadrature(T, n_node):
    gz, gw = gauss_hermite(T["Kz"], n_node)
    return float(sum(gw[i] * integrand(T, gz[i], T["per_cat"]) for i in range(len(gz))))


# ------------------------------------------------------------------ mode and curvature


def inclusion_given_z(T, z):
    """P(j in S | z) for every j, and the per-category count law."""
    b_t = T["b"] - 0.5 * (T["phi"] ** 2).sum(1)
    w = np.exp(b_t + T["phi"] @ z)
    pi = np.zeros(T["J"])
    counts = []
    for c, idx in enumerate(T["cats"]):
        wc, N = w[idx], len(idx)
        a = np.exp(-T["rho"][c] * np.arange(N + 1) * (np.arange(N + 1) - 1) / 2.0)
        e = esp(wc, N)
        pr = a * e
        pr = pr / pr.sum()
        counts.append(pr)
        for pos in range(N):
            keep = np.delete(wc, pos)
            e_m = esp(keep, N - 1)                       # e_r of w without j
            # P(j in S) = sum_r P(r) * w_j e_{r-1}(w_-j) / e_r(w)
            num = sum(pr[r] * wc[pos] * e_m[r - 1] / e[r]
                      for r in range(1, N + 1) if e[r] > 0)
            pi[idx[pos]] = num
    return pi, counts


def find_mode(T, iters=200):
    z = np.zeros(T["Kz"])
    for _ in range(iters):
        pi, _ = inclusion_given_z(T, z)
        z = T["phi"].T @ pi
    return z


def stability(T, z):
    """Lambda(z) = Cov(sum_j phi_j x_j | z), the matrix whose top eigenvalue must be < 1."""
    b_t = T["b"] - 0.5 * (T["phi"] ** 2).sum(1)
    w = np.exp(b_t + T["phi"] @ z)
    L = np.zeros((T["Kz"], T["Kz"]))
    for c, idx in enumerate(T["cats"]):
        N = len(idx)
        a = np.exp(-T["rho"][c] * np.arange(N + 1) * (np.arange(N + 1) - 1) / 2.0)
        e = esp(w[idx], N)
        pr = a * e
        pr = pr / pr.sum()
        # exact within-category second moment by enumerating the category's subsets
        M = np.zeros((T["Kz"], T["Kz"]))
        mu = np.zeros(T["Kz"])
        tot = 0.0
        for r in range(N + 1):
            for sub in itertools.combinations(range(N), r):
                if r == 0:
                    p = pr[0]
                    v = np.zeros(T["Kz"])
                else:
                    p = pr[r] * np.prod(w[idx][list(sub)]) / e[r]
                    v = T["phi"][idx[list(sub)]].sum(0)
                M += p * np.outer(v, v)
                mu += p * v
                tot += p
        L += M - np.outer(mu, mu)
    return L


# ----------------------------------------------------------------------- sampling


def sample_exactly_r(w, r, rng):
    """Exact O(N r) draw of r items with probability proportional to the product of
    weights, by suffix elementary symmetric polynomials."""
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


def sample_basket(T, rng, gz, gw, integ):
    """One exact draw: pick z from its posterior on the quadrature grid, then each
    category independently -- legitimate because conditioning on z is exactly what
    removes the coupling between categories."""
    p = gw * integ
    z = gz[rng.choice(len(gz), p=p / p.sum())]
    b_t = T["b"] - 0.5 * (T["phi"] ** 2).sum(1)
    w = np.exp(b_t + T["phi"] @ z)
    S = np.zeros(T["J"], dtype=bool)
    for c, idx in enumerate(T["cats"]):
        wc, N = w[idx], len(idx)
        a = np.exp(-T["rho"][c] * np.arange(N + 1) * (np.arange(N + 1) - 1) / 2.0)
        e = esp(wc, N)
        pr = a * e
        r = rng.choice(N + 1, p=pr / pr.sum())
        for pos in sample_exactly_r(wc, int(r), rng):
            S[idx[pos]] = True
    return S


# --------------------------------------------------------------------------- main


def run_case(name, T, n_node, rng, n_draws):
    masks, logits = enumerate_all(T)
    M = logits.max()
    Z_bf = float(np.exp(logits - M).sum()) * math.exp(M)
    Z_q = Z_quadrature(T, n_node)
    rel = abs(Z_q - Z_bf) / Z_bf
    log(f"{name}: J={T['J']} ({T['n_cat']}x{T['per_cat']}), Kz={T['Kz']}, "
        f"2^J={2 ** T['J']:,} subsets")
    log(f"    log Z  brute force {math.log(Z_bf):+.10f}   formula {math.log(Z_q):+.10f}"
        f"   rel err {rel:.3e}")

    # 2 -- the empty basket
    p_empty_bf = float(np.exp(logits[0] - M)) * math.exp(M) / Z_bf
    log(f"    P(empty)  brute force {p_empty_bf:.10f}   1/Z {1.0 / Z_q:.10f}"
        f"   rel err {abs(1.0 / Z_q - p_empty_bf) / p_empty_bf:.3e}")

    # 3 -- inclusion probabilities
    p = np.exp(logits - M)
    p /= p.sum()
    pi_bf = (p[:, None] * masks).sum(0)
    gz, gw = gauss_hermite(T["Kz"], n_node)
    integ = np.array([integrand(T, z, T["per_cat"]) for z in gz])
    post = gw * integ
    post /= post.sum()
    pi_f = np.zeros(T["J"])
    for i, z in enumerate(gz):
        pi_f += post[i] * inclusion_given_z(T, z)[0]
    log(f"    inclusion P(j in S): max abs err {np.abs(pi_f - pi_bf).max():.3e} "
        f"over {T['J']} products (mean level {pi_bf.mean():.4f})")

    # 4 -- within-category count law
    co = cat_of(T)
    n0 = masks[:, co == 0].sum(1)
    cnt_bf = np.array([p[n0 == r].sum() for r in range(T["per_cat"] + 1)])
    cnt_f = np.zeros(T["per_cat"] + 1)
    for i, z in enumerate(gz):
        cnt_f += post[i] * inclusion_given_z(T, z)[1][0]
    log(f"    count law of category 0: max abs err {np.abs(cnt_f - cnt_bf).max():.3e}"
        f"   (law {np.round(cnt_bf, 4)})")

    # 5 -- mode and stability
    zh = find_mode(T)
    pi_h, _ = inclusion_given_z(T, zh)
    resid = np.abs(zh - T["phi"].T @ pi_h).max()
    lam = np.linalg.eigvalsh(stability(T, zh)).max()
    log(f"    mode: fixed-point residual {resid:.3e}   lambda_max(Lambda) {lam:.4f}"
        f"   {'stable' if lam < 1 else 'UNSTABLE'}")

    # 6 -- the sampler
    key = {tuple(np.flatnonzero(m)): k for k, m in enumerate(masks)}
    emp = np.zeros(len(masks))
    for _ in range(n_draws):
        emp[key[tuple(np.flatnonzero(sample_basket(T, rng, gz, gw, integ)))]] += 1
    emp /= n_draws
    tv = 0.5 * np.abs(emp - p).sum()
    # noise floor: TV between p and a multinomial sample of the same size drawn FROM p
    floor = np.mean([0.5 * np.abs(rng.multinomial(n_draws, p) / n_draws - p).sum()
                     for _ in range(12)])
    log(f"    sampler: total variation vs enumerated law {tv:.5f}"
        f"   against a sampling-noise floor of {floor:.5f}"
        f"   {'PASS' if tv < 2.0 * floor else 'FAIL'}")
    return rel


def main(a):
    rng = np.random.default_rng(a.seed)
    worst = 0.0
    cases = [
        ("weak coupling ", dict(n_cat=4, per_cat=3, Kz=2, phi_scale=0.25,
                                rho_scale=0.4, b_loc=-1.0)),
        ("strong coupling", dict(n_cat=4, per_cat=3, Kz=2, phi_scale=0.55,
                                 rho_scale=1.2, b_loc=-0.5)),
        ("dense baskets ", dict(n_cat=3, per_cat=4, Kz=2, phi_scale=0.40,
                                rho_scale=0.8, b_loc=+0.3)),
        ("one category  ", dict(n_cat=1, per_cat=8, Kz=2, phi_scale=0.35,
                                rho_scale=1.5, b_loc=-0.3)),
        ("3-d latent    ", dict(n_cat=4, per_cat=3, Kz=3, phi_scale=0.35,
                                rho_scale=0.6, b_loc=-0.7)),
    ]
    for name, cfg in cases:
        T = make_instance(rng, **cfg)
        n_node = a.nodes if cfg["Kz"] <= 2 else max(12, a.nodes // 2)
        worst = max(worst, run_case(name, T, n_node, rng, a.draws))
        print()
    log(f"worst relative error in Z across {len(cases)} cases: {worst:.3e}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--nodes", type=int, default=40)
    p.add_argument("--draws", type=int, default=40000)
    p.add_argument("--seed", type=int, default=0)
    main(p.parse_args())
