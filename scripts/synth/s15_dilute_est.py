"""The decisive test: at REAL dilution, and at the phi'phi grocery actually needs, does the
sampled estimator recover the pair -- or does it saturate?

s14 established that at the real data's pi (0.00114) the pair lift equals exp(phi'phi) to
within 0.3%.  So the 2.5x co-occurrence in grocery needs phi'phi = ln(2.5) = 0.92, not the
2-4 the dense benches implied.  s13 found the anisotropic proposal works at phi'phi = 1 and
collapses at 2 -- but s13 was run at pi = 0.30, thirty times denser than reality.

Both the required strength and the estimator's breaking point are regime-dependent, and s13
measured them in different regimes.  This script measures them in the SAME one: the dilute
configuration from s14, at phi'phi = 0.92.

Two practical constraints shape the design.

  Rarity costs data.  At pi = 0.0507 with lift 2.19, a pair co-occurs in 0.0507^2 * 2.19 =
  0.56% of baskets, so 200,000 baskets give ~1,100 co-occurrences to learn from.  Pushing to
  pi = 0.0025 would leave ~1 and there would be nothing to fit.  pi = 0.0507 already reaches
  87% of the dilute limit, so it is dilute enough to be informative and dense enough to be
  learnable -- that is the whole reason for this operating point.

  The cap must not be the thing under test.  It is set at 1.5x the planted norm, the same
  slack s7 and s13 used, so a failure is the estimator saturating rather than the cap binding.

Run:  python3 s15_dilute_est.py
"""
import math
import time

import numpy as np
import torch

torch.set_default_dtype(torch.float64)

J = 20
KZ = 12
SIZE = 6.0
NB = 200000
STEPS = 900
DRAWS = 256
B_PAIR = -3.0           # gives pi ~ 0.051, 87% of the dilute limit, still learnable
TRUE_T = 0.92           # ln(2.5): the strength grocery co-occurrence actually needs


def build_mask(Jd):
    idx = torch.arange(2 ** Jd, dtype=torch.int64)
    return ((idx.unsqueeze(1) >> torch.arange(Jd, dtype=torch.int64)) & 1).to(torch.float64)


def energy(mask, b, PH):
    v = mask @ PH
    return mask @ b + 0.5 * ((v * v).sum(1) - mask @ (PH ** 2).sum(1))


def report(mask_ne, E):
    p = torch.softmax(E, 0)
    pi = (mask_ne * p.unsqueeze(1)).sum(0)
    size = float((mask_ne.sum(1) * p).sum())
    lift = {}
    for (a, c) in ((0, 1), (0, 2)):
        joint = float((mask_ne[:, a] * mask_ne[:, c] * p).sum())
        lift[(a, c)] = joint / max(float(pi[a] * pi[c]), 1e-300)
    return pi, size, lift


def fill_for_size(mask_ne, PH, target):
    lo, hi = -14.0, 8.0
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        b = torch.full((J,), mid)
        b[0] = b[1] = b[2] = B_PAIR
        _, s, _ = report(mask_ne, energy(mask_ne, b, PH))
        if s < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def mode_and_vtop(b, PH, steps=6):
    z = torch.zeros(1, KZ)
    for _ in range(steps):
        zz = z.detach().requires_grad_(True)
        w = torch.exp(b + zz @ PH.T - 0.5 * (PH ** 2).sum(1))
        z = torch.autograd.grad(torch.log1p(w).sum(), zz)[0]
    zh = z.detach()
    w = torch.exp(b + zh @ PH.T - 0.5 * (PH ** 2).sum(1))[0]
    pi = (w / (1 + w)).clamp(1e-12, 1 - 1e-12)
    L = (PH.detach() * (pi * (1 - pi)).unsqueeze(1)).T @ PH.detach()
    _, V = torch.linalg.eigh(L)
    return zh, V[:, -1]


def logZ_single(b, PH, nd, gen):
    zh, _ = mode_and_vtop(b, PH)
    eps = torch.randn(nd, KZ, generator=gen)
    zs = zh + eps
    w = torch.exp(b.unsqueeze(0) + zs @ PH.T - 0.5 * (PH ** 2).sum(1).unsqueeze(0))
    lf = torch.expm1(torch.log1p(w).sum(1)).clamp_min(1e-300).log()
    lw = (-0.5 * (zs ** 2).sum(1) + lf) - (-0.5 * eps.pow(2).sum(1))
    return torch.logsumexp(lw, 0) - math.log(nd)


def logZ_aniso(b, PH, nd, gen, s=2.0):
    zh, vtop = mode_and_vtop(b, PH)
    e0 = torch.randn(nd, KZ, generator=gen)
    proj = e0 @ vtop
    eps = e0 + (s - 1.0) * proj.unsqueeze(1) * vtop.unsqueeze(0)
    r = eps @ vtop
    perp = eps - r.unsqueeze(1) * vtop.unsqueeze(0)
    lq = -0.5 * (perp.pow(2).sum(1) + (r / s) ** 2) - math.log(s)
    zs = zh + eps
    w = torch.exp(b.unsqueeze(0) + zs @ PH.T - 0.5 * (PH ** 2).sum(1).unsqueeze(0))
    lf = torch.expm1(torch.log1p(w).sum(1)).clamp_min(1e-300).log()
    lw = (-0.5 * (zs ** 2).sum(1) + lf) - lq
    return torch.logsumexp(lw, 0) - math.log(nd)


def main():
    mask = build_mask(J)
    mask_ne = mask[mask.sum(1) > 0].contiguous()
    del mask

    v = math.sqrt(TRUE_T)
    PH_t = torch.zeros(J, KZ)
    PH_t[0, 0] = v
    PH_t[1, 0] = v
    bf = fill_for_size(mask_ne, PH_t, SIZE)
    b_t = torch.full((J,), bf)
    b_t[0] = b_t[1] = b_t[2] = B_PAIR
    E_t = energy(mask_ne, b_t, PH_t)
    pi_t, size_t, lift_t = report(mask_ne, E_t)

    print(f"J = {J}, Kz = {KZ}, mean size {size_t:.2f}, pair marginal pi = {float(pi_t[0]):.5f}")
    print(f"planted phi'phi = {TRUE_T:.3f} (= ln 2.5)   true lift = {lift_t[(0,1)]:.3f}"
          f"   control = {lift_t[(0,2)]:.3f}")

    rng = np.random.default_rng(0)
    p_t = torch.softmax(E_t, 0).numpy()
    draw = rng.choice(mask_ne.shape[0], size=NB, p=p_t)
    uniq, inv = np.unique(draw, return_inverse=True)
    red = mask_ne[torch.as_tensor(uniq)].contiguous()
    cnt = torch.as_tensor(np.bincount(inv).astype(np.float64))
    both = float((red[:, 0] * red[:, 1] * cnt).sum())
    print(f"{NB:,} baskets, {len(uniq):,} distinct; the pair co-occurs in "
          f"{int(both):,} of them ({100*both/NB:.2f}%)\n")

    cap = 1.5 * v
    print(f"norm cap {cap:.3f} (cap^2 = {cap**2:.3f}, so phi'phi can reach "
          f"{cap**2:.2f} -- the cap is not the constraint under test)")
    print(f"{'mode':>8}{'phi0.phi1':>11}{'lift':>9}{'control':>9}{'size':>7}{'time':>7}")
    for mode in ("exact", "single", "aniso"):
        t0 = time.time()
        bh = torch.full((J,), -1.0, requires_grad=True)
        PH = (torch.randn(J, KZ, generator=torch.Generator().manual_seed(1))
              * 0.1).requires_grad_(True)
        opt = torch.optim.Adam([bh, PH], lr=0.05)
        gen = torch.Generator().manual_seed(7)
        for step in range(STEPS):
            Ed = energy(red, bh, PH)
            if mode == "exact":
                lz = torch.logsumexp(energy(mask_ne, bh, PH), 0)
            elif mode == "single":
                lz = logZ_single(bh, PH, DRAWS, gen)
            else:
                lz = logZ_aniso(bh, PH, DRAWS, gen)
            ll = ((Ed - lz) * cnt).sum() / cnt.sum()
            opt.zero_grad()
            (-ll).backward()
            opt.step()
            with torch.no_grad():
                nn_ = PH.norm(dim=1, keepdim=True).clamp_min(1e-12)
                PH.mul_((cap / nn_).clamp(max=1.0))
        with torch.no_grad():
            _, szf, lff = report(mask_ne, energy(mask_ne, bh, PH))
            print(f"{mode:>8}{float(PH[0]@PH[1]):11.3f}{lff[(0,1)]:9.3f}"
                  f"{lff[(0,2)]:9.3f}{szf:7.2f}{time.time()-t0:6.0f}s", flush=True)
    print(f"\n{'truth':>8}{TRUE_T:11.3f}{lift_t[(0,1)]:9.3f}{lift_t[(0,2)]:9.3f}{size_t:7.2f}")


if __name__ == "__main__":
    main()
