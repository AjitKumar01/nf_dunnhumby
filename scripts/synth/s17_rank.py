"""Every real run collapses to erank 2.  Is that the estimator, or the data?

The effective rank of phi has sat at 2 in run39, run49, run54 and run55 alike, across three
different objectives.  A rank-2 phi cannot encode many distinct complementarity groups: two
directions give essentially one global "big basket" axis plus one contrast, so every pair's
lift is forced to be a function of those two numbers.  Real grocery needs many independent
groups -- pasta/sauce has nothing to do with shampoo/conditioner.

No bench so far has tested this.  s3, s7, s12, s13, s15 and s16 all plant ONE complementary
pair (s3 adds one substitutable pair).  Rank-1 truth cannot reveal a rank collapse, so the
benches were structurally blind to the one symptom every real run shows.

Here the truth is r disjoint pairs, each in its own orthogonal latent direction, each at the
phi'phi = 0.92 that grocery needs.  The planted phi therefore has rank exactly r.  The
question is whether the fit returns rank r or squashes it.

erank is reported as exp(entropy of the normalised singular value spectrum), the same measure
fit.py logs, so the numbers are directly comparable to the real runs.

Run:  python3 s17_rank.py
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
B_PAIR = -3.0
TRUE_T = 0.92


def build_mask(Jd):
    idx = torch.arange(2 ** Jd, dtype=torch.int64)
    return ((idx.unsqueeze(1) >> torch.arange(Jd, dtype=torch.int64)) & 1).to(torch.float64)


def energy(mask, b, PH):
    v = mask @ PH
    return mask @ b + 0.5 * ((v * v).sum(1) - mask @ (PH ** 2).sum(1))


def erank(PH):
    """Participation ratio (sum s^2)^2 / sum s^4 -- exactly what fit.py logs.

    Not the entropy of the spectrum: entropy counts a long tail of near-zero directions
    almost as heavily as the real ones, and reported erank ~9 for a rank-1 truth.  The
    participation ratio is dominated by the large singular values, which is why fit.py uses
    it and why "erank 2" in the real runs means two directions carry the mass.
    """
    s = torch.linalg.svdvals(PH)
    return float((s ** 2).sum() ** 2 / (s ** 4).sum().clamp_min(1e-30))


def erank_H(PH):
    """The entropy version, kept alongside so the two measures can be compared directly."""
    s = torch.linalg.svdvals(PH)
    p = (s / s.sum().clamp_min(1e-30)).clamp_min(1e-30)
    return float(torch.exp(-(p * p.log()).sum()))


def stats(mask_ne, b, PH, pairs):
    E = energy(mask_ne, b, PH)
    p = torch.softmax(E, 0)
    pi = (mask_ne * p.unsqueeze(1)).sum(0)
    size = float((mask_ne.sum(1) * p).sum())
    lifts = []
    for (a, c) in pairs:
        joint = float((mask_ne[:, a] * mask_ne[:, c] * p).sum())
        lifts.append(joint / max(float(pi[a] * pi[c]), 1e-300))
    return pi, size, lifts


def calib_b(mask_ne, PH, npair, target, iters=34):
    lo, hi = -14.0, 8.0
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        b = torch.full((J,), mid)
        b[:2 * npair] = B_PAIR
        _, s, _ = stats(mask_ne, b, PH, [])
        if s < target:
            lo = mid
        else:
            hi = mid
    b = torch.full((J,), 0.5 * (lo + hi))
    b[:2 * npair] = B_PAIR
    return b


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
    print(f"J = {J}, Kz = {KZ}, mean size {SIZE}, each planted pair at phi'phi = {TRUE_T} "
          f"(= ln 2.5) in its own\northogonal direction, so the planted phi has rank exactly "
          f"r.  Real runs sit at erank 2.\n")
    print(f"{'r':>3}{'er true':>9}{'L true':>8} | {'er exact':>9}{'er aniso':>9} | "
          f"{'d exact':>9}{'d aniso':>9} | {'L exact':>9}{'L aniso':>9} | "
          f"{'worst ex':>9}{'worst an':>9}{'Hex':>8}{'Han':>8}{'time':>7}")

    v = math.sqrt(TRUE_T)
    for r in (1, 2, 4, 6):
        t0 = time.time()
        pairs = [(2 * i, 2 * i + 1) for i in range(r)]
        PH_t = torch.zeros(J, KZ)
        for i, (a, c) in enumerate(pairs):
            PH_t[a, i] = v
            PH_t[c, i] = v                      # direction i is used by pair i alone
        b_t = calib_b(mask_ne, PH_t, r, SIZE)
        _, size_t, lift_t = stats(mask_ne, b_t, PH_t, pairs)

        rng = np.random.default_rng(0)
        p_t = torch.softmax(energy(mask_ne, b_t, PH_t), 0).numpy()
        draw = rng.choice(mask_ne.shape[0], size=NB, p=p_t)
        uniq, inv = np.unique(draw, return_inverse=True)
        red = mask_ne[torch.as_tensor(uniq)].contiguous()
        cnt = torch.as_tensor(np.bincount(inv).astype(np.float64))

        cap = 1.5 * v
        res = {}
        for mode in ("exact", "aniso"):
            bh = torch.full((J,), -1.0, requires_grad=True)
            PH = (torch.randn(J, KZ, generator=torch.Generator().manual_seed(1))
                  * 0.1).requires_grad_(True)
            opt = torch.optim.Adam([bh, PH], lr=0.05)
            gen = torch.Generator().manual_seed(7)
            for step in range(STEPS):
                Ed = energy(red, bh, PH)
                lz = (torch.logsumexp(energy(mask_ne, bh, PH), 0) if mode == "exact"
                      else logZ_aniso(bh, PH, DRAWS, gen))
                ll = ((Ed - lz) * cnt).sum() / cnt.sum()
                opt.zero_grad()
                (-ll).backward()
                opt.step()
                with torch.no_grad():
                    nn_ = PH.norm(dim=1, keepdim=True).clamp_min(1e-12)
                    PH.mul_((cap / nn_).clamp(max=1.0))
            with torch.no_grad():
                _, _, lf = stats(mask_ne, bh, PH, pairs)
                dots = [float(PH[a] @ PH[c]) for (a, c) in pairs]
                res[mode] = (erank(PH.detach()), float(np.mean(dots)),
                             float(np.mean(lf)), float(np.min(lf)),
                             erank_H(PH.detach()))

        print(f"{r:3d}{erank(PH_t):9.2f}{float(np.mean(lift_t)):8.3f} | "
              f"{res['exact'][0]:9.2f}{res['aniso'][0]:9.2f} | "
              f"{res['exact'][1]:9.3f}{res['aniso'][1]:9.3f} | "
              f"{res['exact'][2]:9.3f}{res['aniso'][2]:9.3f} | "
              f"{res['exact'][3]:9.3f}{res['aniso'][3]:9.3f}"
              f"{res['exact'][4]:8.2f}{res['aniso'][4]:8.2f}"
              f"{time.time()-t0:6.0f}s", flush=True)


if __name__ == "__main__":
    main()
