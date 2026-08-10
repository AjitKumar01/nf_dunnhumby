"""
Exact sampling from P(S | K) proportional to exp E(S).

The current model cannot do this.  It draws a first basket from independent within-category
softmaxes -- which ignores the interaction entirely -- and then runs four Gibbs sweeps and
hopes.  Nothing checks whether four is enough.

With Z computable, exact sampling falls out of the same identity, because the auxiliary
Gaussian variable z makes the categories conditionally independent:

    P(S) = E_z [ prod_c (choose k_c from c with weights w_j(z)) ] / Z

so sampling is two steps:

    1.  draw z from its posterior  q(z) proportional to N(z;0,I) * prod_c e_{k_c}(w(z))
    2.  given z, draw each category INDEPENDENTLY, choosing exactly k_c items with
        weights w_j(z)

Step 1 is done by sampling-importance-resampling from the same Laplace proposal used for
log Z, which is efficient here because its effective sample size is around 0.7.

Step 2 is the classical sequential algorithm for drawing exactly k items with given
weights.  Walking the items in order and keeping suffix elementary symmetric polynomials,

    P(item i is taken | m still needed from items i..N) = w_i e_{m-1}(w_{i+1..N})
                                                          / e_m(w_i..w_N)

which is exact, costs O(N k), and needs no rejection.

No Gibbs, no burn-in, no mixing diagnostic.
"""
import numpy as np


def suffix_esp(w, kmax):
    """suf[i][m] = e_m(w_i, ..., w_{N-1}).  Shape [N+1, kmax+1]."""
    N = len(w)
    suf = np.zeros((N + 1, kmax + 1))
    suf[N, 0] = 1.0
    for i in range(N - 1, -1, -1):
        suf[i] = suf[i + 1]
        suf[i, 1:] = suf[i + 1, 1:] + w[i] * suf[i + 1, :-1]
    return suf


def sample_exactly_k(w, k, rng):
    """Draw exactly k of the N items, with probability proportional to the product of the
    chosen weights.  Exact, O(N k), no rejection."""
    if k == 0:
        return []
    N = len(w)
    suf = suffix_esp(w, k)
    out, need = [], k
    for i in range(N):
        if need == 0:
            break
        if N - i == need:                       # everything left must be taken
            out.extend(range(i, N))
            break
        denom = suf[i, need]
        if denom <= 0:
            out.extend(range(i, i + need))
            break
        p = w[i] * suf[i + 1, need - 1] / denom
        if rng.random() < p:
            out.append(i)
            need -= 1
    return out


def log_integrand(z, cats_c, cats_phi, ks, lam):
    t = 0.0
    for c, (cj, phi, k) in enumerate(zip(cats_c, cats_phi, ks)):
        x = cj + np.sqrt(lam) * (phi @ z)
        M = x.max()
        suf = suffix_esp(np.exp(x - M), k)
        t += np.log(max(suf[0, k], 1e-300)) + k * M
    return t


def sample_basket(cats_c, cats_phi, ks, lam, rng, n_prop=256, zh=None, sd=None):
    """One exact draw from P(S | K).

    z is drawn from its posterior by sampling-importance-resampling: propose from the
    Laplace Gaussian, weight by target/proposal, and resample one.  Then each category is
    drawn independently and exactly, which is legitimate because conditioning on z is
    exactly what removes the coupling between categories.
    """
    K = cats_phi[0].shape[1]
    if zh is None:
        zh = np.zeros(K)
        for _ in range(40):                                    # fixed-point mode
            acc = np.zeros(K)
            for cj, phi, k in zip(cats_c, cats_phi, ks):
                x = cj + np.sqrt(lam) * (phi @ zh)
                M = x.max()
                w = np.exp(x - M)
                suf = suffix_esp(w, k)
                pre = np.zeros((len(w) + 1, k + 1))
                pre[0, 0] = 1.0
                for i in range(len(w)):
                    pre[i + 1] = pre[i]
                    pre[i + 1, 1:] = pre[i, 1:] + w[i] * pre[i, :-1]
                ek = pre[len(w), k]
                pi = np.array([w[j] * sum(pre[j, a] * suf[j + 1, k - 1 - a]
                                          for a in range(k)) / max(ek, 1e-300)
                               for j in range(len(w))])
                acc += phi.T @ pi
            zh = np.sqrt(lam) * acc
    if sd is None:
        sd = np.ones(K)
    zs = zh + rng.standard_normal((n_prop, K)) * sd
    lp = np.array([-0.5 * zz @ zz + log_integrand(zz, cats_c, cats_phi, ks, lam)
                   for zz in zs])
    lq = -0.5 * (((zs - zh) / sd) ** 2).sum(1)
    l = lp - lq
    l -= l.max()
    p = np.exp(l)
    p /= p.sum()
    z = zs[rng.choice(len(zs), p=p)]
    S = []
    for cj, phi, k in zip(cats_c, cats_phi, ks):
        x = cj + np.sqrt(lam) * (phi @ z)
        idx = sample_exactly_k(np.exp(x - x.max()), k, rng)
        S.append(sorted(idx))
    return S, zh
