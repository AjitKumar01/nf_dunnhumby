"""Can the size law be recovered at all?  The bench never asked.

s1-s18 validated the interaction: pair lift is monotone in phi'phi, planted +-1.000 comes back
as +0.979/-0.994, and at real dilution lift = exp(phi'phi) to within 0.3%.  But every one of
those tests set rho_0 = 0, and s12 checked basket size by its MEAN alone (`sz fit`).  So the
120-parameter size function that all of the real trouble runs through was never exercised, and
the bench was structurally incapable of catching a size-law failure.  That is why it kept
giving clean answers about phi while the real runs failed on E[n] and Var(n).

This asks the question directly, with log Z summed exactly over all 2^J subsets so that any
failure is the objective or the optimiser, never the estimator:

    plant a known rho_0 -> generate baskets -> refit b, phi and rho_0 -> compare P(n) by KL

Compared by KL and TV, not by the mean.  Two laws can share a mean and be entirely different
distributions -- on the real held-out sizes a two-point law matched to both moments scores
KL 554 while passing the mean and variance goals fit.py used all session.

IDENTIFIABILITY.  rho_0 is only determined up to an affine function of n: adding c*n to rho_0
and c to every b_j leaves E(S) unchanged, since sum_{j in S} b_j moves by c*|S| exactly as
rho_0 does.  rho_0(0) is likewise free.  So the recovered rho_0 CANNOT match the planted one
coefficient by coefficient, and comparing them directly would manufacture a failure.  P(n) is
invariant to that gauge, which is the other reason to score the distribution rather than the
parameters.

Three fits per configuration, to separate the possible causes:
    free    b, phi and rho_0 all fitted        -- can the law be recovered at all?
    frozen  rho_0 held at 0, b and phi fitted  -- what does rho_0 actually buy?
    aniso   free, but log Z importance-sampled -- does the sampled gradient still get there?

Run:  python3 s19_sizelaw.py
"""
import math
import time

import numpy as np
import torch

torch.set_default_dtype(torch.float64)

J = 16
K = 4
KZ = 12
NB = 200000
STEPS = 1500
DRAWS = 256


def build_mask(Jd):
    idx = torch.arange(2 ** Jd, dtype=torch.int64)
    return ((idx.unsqueeze(1) >> torch.arange(Jd, dtype=torch.int64)) & 1).to(torch.float64)


def energy(mask, nvec, b, PH, r0):
    """E(S) = sum_j b_j + sum_{j<k} phi_j.phi_k - rho_0(|S|)."""
    v = mask @ PH
    pair = 0.5 * ((v * v).sum(1) - mask @ (PH ** 2).sum(1))
    return mask @ b + pair - r0[nvec]


def size_law(mask_ne, nvec_ne, E, Jd):
    p = torch.softmax(E, 0)
    law = torch.zeros(Jd + 1, dtype=p.dtype)
    law = law.index_add(0, nvec_ne, p)
    return law[1:] / law[1:].sum()          # conditioned on non-empty, as the model is


def stats(law):
    k = torch.arange(1, law.shape[0] + 1, dtype=law.dtype)
    m = float((k * law).sum())
    return m, float((k * k * law).sum() - m * m)


def kl_tv(p_fit, p_true):
    m = p_true > 1e-12
    kl = float((p_true[m] * (p_true[m] / p_fit[m].clamp_min(1e-300)).log()).sum())
    return kl, float(0.5 * (p_fit - p_true).abs().sum())


def logZ_aniso(b, PH, r0, nd, gen, Jd, s=2.0):
    kz = PH.shape[1]        # latent width IS phi's width; KZ is not a free choice here
    """The same proposal ragged.py uses, with rho_0 inside the latent factorisation."""
    z = torch.zeros(1, kz)
    for _ in range(4):
        zz = z.detach().requires_grad_(True)
        w = torch.exp(b + zz @ PH.T - 0.5 * (PH ** 2).sum(1))
        z = torch.autograd.grad(torch.log1p(w).sum(), zz)[0]
    zh = z.detach()
    w0 = torch.exp(b + zh @ PH.T - 0.5 * (PH ** 2).sum(1))[0]
    pi = (w0 / (1 + w0)).clamp(1e-12, 1 - 1e-12)
    L = (PH.detach() * (pi * (1 - pi)).unsqueeze(1)).T @ PH.detach()
    _, V = torch.linalg.eigh(L)
    vtop = V[:, -1]
    e0 = torch.randn(nd, kz, generator=gen)
    eps = e0 + (s - 1.0) * (e0 @ vtop).unsqueeze(1) * vtop.unsqueeze(0)
    r = eps @ vtop
    perp = eps - r.unsqueeze(1) * vtop.unsqueeze(0)
    lq = -0.5 * (perp.pow(2).sum(1) + (r / s) ** 2) - math.log(s)
    zs = zh + eps
    lw_item = b.unsqueeze(0) + zs @ PH.T - 0.5 * (PH ** 2).sum(1).unsqueeze(0)
    # elementary symmetric polynomial over items, so rho_0 can weight each size
    A = torch.zeros(nd, Jd + 1, dtype=b.dtype)
    A[:, 0] = 1.0
    for j in range(Jd):
        wj = torch.exp(lw_item[:, j]).unsqueeze(1)
        A = A + torch.nn.functional.pad(A[:, :-1], (1, 0)) * wj
    lf = torch.logsumexp(A.clamp_min(1e-300).log() - r0.unsqueeze(0), dim=1)
    lw = (-0.5 * (zs ** 2).sum(1) + lf) - lq
    return torch.logsumexp(lw, 0) - math.log(nd)


def main():
    mask = build_mask(J)
    nvec = mask.sum(1).long()
    ne = nvec > 0
    mask_ne, nvec_ne = mask[ne].contiguous(), nvec[ne]
    del mask
    g = torch.Generator().manual_seed(0)

    print(f"J = {J}, log Z summed over all {2**J:,} subsets, {NB:,} baskets, {STEPS} steps")
    print("rho_0 is identifiable only up to an affine function of n, so P(n) is scored, "
          "not the coefficients.\n")
    print(f"{'planted':>26} | {'fit':>7} | {'mean':>6}{'var':>7}{'v/m':>6} | "
          f"{'KL':>8}{'TV':>7}{'time':>7}")

    for tag, curve in (("flat  (rho_0 = 0)", lambda n: 0.0 * n),
                       ("narrow(+0.05 n^2)", lambda n: 0.05 * n ** 2),
                       ("wide  (-0.02 n^2)", lambda n: -0.02 * n ** 2),
                       ("bumpy (sin)", lambda n: 1.5 * torch.sin(n * 0.9))):
        nn = torch.arange(J + 1, dtype=torch.float64)
        r0_t = curve(nn)
        r0_t = r0_t - r0_t[0]
        b_t = torch.full((J,), -1.0)
        PH_t = torch.zeros(J, K)
        PH_t[0, 0] = 1.0
        PH_t[1, 0] = 1.0                       # one complementary pair, as in s3
        E_t = energy(mask_ne, nvec_ne, b_t, PH_t, r0_t)
        law_t = size_law(mask_ne, nvec_ne, E_t, J)
        mt, vt = stats(law_t)
        print(f"{tag:>26} | {'TRUTH':>7} | {mt:6.2f}{vt:7.2f}{vt/mt:6.2f} | "
              f"{0.0:8.4f}{0.0:7.3f}")

        p_t = torch.softmax(E_t, 0).numpy()
        draw = np.random.default_rng(0).choice(mask_ne.shape[0], size=NB, p=p_t)
        uniq, inv = np.unique(draw, return_inverse=True)
        red = mask_ne[torch.as_tensor(uniq)].contiguous()
        redn = nvec_ne[torch.as_tensor(uniq)]
        cnt = torch.as_tensor(np.bincount(inv).astype(np.float64))

        for mode in ("free", "frozen", "aniso"):
            t0 = time.time()
            bh = torch.full((J,), -0.5, requires_grad=True)
            PH = (torch.randn(J, K, generator=torch.Generator().manual_seed(1))
                  * 0.1).requires_grad_(True)
            r0 = torch.zeros(J + 1, requires_grad=(mode != "frozen"))
            params = [bh, PH] + ([r0] if mode != "frozen" else [])
            opt = torch.optim.Adam(params, lr=0.05)
            gen = torch.Generator().manual_seed(7)
            for _ in range(STEPS):
                Ed = energy(red, redn, bh, PH, r0)
                if mode == "aniso":
                    lz = logZ_aniso(bh, PH, r0, DRAWS, gen, J)
                else:
                    lz = torch.logsumexp(energy(mask_ne, nvec_ne, bh, PH, r0), 0)
                ll = ((Ed - lz) * cnt).sum() / cnt.sum()
                opt.zero_grad()
                (-ll).backward()
                opt.step()
            with torch.no_grad():
                law_f = size_law(mask_ne, nvec_ne,
                                 energy(mask_ne, nvec_ne, bh, PH, r0), J)
                mf, vf = stats(law_f)
                kl, tv = kl_tv(law_f, law_t)
            print(f"{'':>26} | {mode:>7} | {mf:6.2f}{vf:7.2f}{vf/mf:6.2f} | "
                  f"{kl:8.4f}{tv:7.3f}{time.time()-t0:6.0f}s", flush=True)
        print()


if __name__ == "__main__":
    main()
