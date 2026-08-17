"""
Validate the version-3 implementation against explicit enumeration.

Nothing here is a unit test of style; each check is a claim from paper/version_3.html
evaluated on a catalogue small enough to enumerate every subset:

  1  the energy E(S), against a from-scratch loop over the definition in Eq. 8
  2  log Z from Theorem 1's importance sampler, against sum over all 2^J subsets
  3  P(empty) = 1/Z, Corollary 1
  4  the likelihood normalises: sum over all NON-EMPTY subsets of exp(loglik) = 1
  5  the three-level sampler of Eq. 18b, against the enumerated law
  6  the gradient of log Z equals the model's expected purchase vector, Eq. 17

Run:  python3 validate.py
"""
import argparse
import itertools
import math

import torch

from core import Model, sample


def log(m):
    print(f"[val] {m}", flush=True)


def brute(model, house):
    """Every subset, with E(S) computed from the definition rather than from
    model.energy -- so a bug in model.energy shows up rather than cancelling."""
    J, C, P = model.J, model.C, model.P
    b = model.b(house)[0].detach()
    phi = model.phi.detach()
    rho_c = model.rho_c.detach()
    rho_0 = model.rho_0().detach()
    Es, masks = [], []
    for bits in itertools.product([0, 1], repeat=J):
        idx = [j for j, v in enumerate(bits) if v]
        E = sum(float(b[j]) for j in idx)
        for a in range(len(idx)):
            for c in range(a + 1, len(idx)):
                E += float(phi[idx[a]] @ phi[idx[c]])
        for c in range(C):
            nc = sum(1 for j in idx if j // P == c)
            E -= float(rho_c[c]) * nc * (nc - 1) / 2.0
        n = len(idx)
        E -= float(rho_0[min(n, model.nmax)])
        Es.append(E)
        masks.append(torch.tensor(bits, dtype=torch.float64))
    return torch.stack(masks), torch.tensor(Es, dtype=torch.float64)


def main(a):
    torch.set_default_dtype(torch.float64)
    g = torch.Generator().manual_seed(a.seed)
    m = Model(J=a.C * a.P, N=4, C=a.C, P=a.P, K=4, Kz=a.Kz, nmax=a.C * a.P,
              R=min(a.P, 4), seed=a.seed)
    with torch.no_grad():                     # give every parameter a non-trivial value
        m.lam.normal_(-0.8, 0.6, generator=g)
        m.phi.normal_(0.0, 0.35, generator=g)
        m.rho_c.normal_(0.0, 0.6, generator=g)
        m.rho_0_free.normal_(0.0, 0.4, generator=g)
    house = torch.tensor([0])
    masks, Es = brute(m, house)
    log(f"catalogue {m.J} products in {m.C} categories, Kz={m.Kz}; "
        f"{2 ** m.J:,} subsets enumerated")

    # 1 -- energy
    mine = m.energy(house.repeat(len(masks)), masks).detach()
    log(f"1  energy E(S): max abs err {float((mine - Es).abs().max()):.3e}")

    # 2 -- log Z
    true_lz = torch.logsumexp(Es, 0)
    lz, ess = m.log_Z(house, n_draws=a.draws, generator=g, return_ess=True)
    log(f"2  log Z: brute force {float(true_lz):+.8f}   sampler {float(lz[0]):+.8f}   "
        f"err {float(lz[0] - true_lz):+.2e}   ESS {float(ess[0]):.3f}")

    # 3 -- the empty basket
    p = torch.softmax(Es, 0)
    log(f"3  P(empty): brute force {float(p[0]):.10f}   1/Z {float(torch.exp(-true_lz)):.10f}"
        f"   rel err {abs(float(p[0]) - float(torch.exp(-true_lz))) / float(p[0]):.2e}")

    # 4 -- the likelihood normalises over non-empty baskets
    nz = masks.sum(-1) > 0
    ll = m.loglik(house.repeat(int(nz.sum())), masks[nz], n_draws=a.draws, generator=g)
    log(f"4  sum over non-empty of exp(loglik) = {float(torch.exp(ll).sum()):.8f}  "
        f"(should be 1)")

    # 5 -- the sampler
    key = {tuple(int(v) for v in row): i for i, row in enumerate(masks)}
    emp = torch.zeros(len(masks))
    for _ in range(a.samples):
        S, _ = sample(m, house, n_draws=64, generator=g)
        emp[key[tuple(int(v) for v in S[0])]] += 1
    emp /= a.samples
    tv = 0.5 * float((emp - p).abs().sum())
    floor = sum(0.5 * float((torch.distributions.Multinomial(a.samples, p).sample()
                             / a.samples - p).abs().sum()) for _ in range(8)) / 8
    log(f"5  sampler: total variation {tv:.5f} against a noise floor of {floor:.5f}   "
        f"{'PASS' if tv < 2 * floor else 'FAIL'}")

    # 6 -- d log Z / d b_j = P(j in S)
    pi_true = (p.unsqueeze(-1) * masks).sum(0)
    lam0 = m.lam.detach().clone()
    m.lam.requires_grad_(True)
    lz2 = m.log_Z(house, n_draws=4096, generator=torch.Generator().manual_seed(7))
    grad = torch.autograd.grad(lz2.sum(), m.lam)[0]
    m.lam.data = lam0
    log(f"6  d log Z / d b_j vs P(j in S): max abs err "
        f"{float((grad - pi_true).abs().max()):.3e}  (mean level {float(pi_true.mean()):.4f})")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--C", type=int, default=3)
    p.add_argument("--P", type=int, default=3)
    p.add_argument("--Kz", type=int, default=2)
    p.add_argument("--draws", type=int, default=8192)
    p.add_argument("--samples", type=int, default=6000)
    p.add_argument("--seed", type=int, default=0)
    main(p.parse_args())
