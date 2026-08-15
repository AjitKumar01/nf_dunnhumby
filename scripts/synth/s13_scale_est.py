"""At the bigger catalogue AND the bigger basket size, does the estimator still break?

s7 showed plain importance sampling returns lift 1.000 at every planted strength -- no pair
structure at all -- while the anisotropic proposal held to about phi'phi = 2 and then failed.
That was at J=12 with a mean basket size of 1.86.  s12 then showed something that changes the
stakes: the SAME planted phi'phi buys much less lift once baskets are big, because a bigger
basket has more competing pairs.  So the real regime needs a LARGER phi'phi to reach the lift
grocery data shows, which pushes it further into the range where the estimator already failed.

This script tests that directly: J = 20, mean size 6, log Z estimated three ways.

  exact   all 2^J subsets summed              -- the control, no estimation
  single  plain IS at the Laplace mode        -- what the real fit used for most of the run
  aniso   IS stretched along the top curvature direction -- the best proposal found

The data term only needs E(S) at the observed subsets, so it runs on a reduced mask of the
distinct baskets drawn; only `exact` pays for the full enumeration, and only for log Z.

Run:  python3 s13_scale_est.py
"""
import math
import time

import numpy as np
import torch

torch.set_default_dtype(torch.float64)

J = 20
KZ = 12             # latent width, matched to the real model's Kz
NB = 40000
STEPS = 900
DRAWS = 256
TARGET_SIZE = 6.0


def build_mask(Jd):
    idx = torch.arange(2 ** Jd, dtype=torch.int64)
    return ((idx.unsqueeze(1) >> torch.arange(Jd, dtype=torch.int64)) & 1).to(torch.float64)


def energy(mask, b, PH):
    v = mask @ PH
    sq = mask @ (PH ** 2).sum(1)
    return mask @ b + 0.5 * ((v * v).sum(1) - sq)


def lift_of(mask_ne, E, pairs):
    p = torch.softmax(E, 0)
    pi = (mask_ne * p.unsqueeze(1)).sum(0)
    size = float((mask_ne.sum(1) * p).sum())
    out = {}
    for (a, c) in pairs:
        joint = float((mask_ne[:, a] * mask_ne[:, c] * p).sum())
        out[(a, c)] = joint / max(float(pi[a] * pi[c]), 1e-300)
    return out, size


def calibrate_b(mask_ne, PH, target):
    lo, hi = -12.0, 4.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        E = energy(mask_ne, torch.full((J,), mid), PH)
        s = float((mask_ne.sum(1) * torch.softmax(E, 0)).sum())
        if s < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ---- the two sampled normalisers, identical in form to ragged.py -----------------------
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
    pairs = [(0, 1), (4, 5), (0, 2)]
    print(f"J = {J}, mean basket size calibrated to {TARGET_SIZE}, Kz = {KZ}, "
          f"{NB:,} baskets, {DRAWS} draws, {STEPS} steps")
    print(f"{'true':>6}{'cap^2':>7}{'L true':>8} | {'exact':>8}{'single':>8}{'aniso':>8} | "
          f"{'L exact':>8}{'L sgl':>8}{'L ani':>8} | {'sz ex':>6}{'sz sg':>6}{'sz an':>6}"
          f"{'time':>7}")

    for t in (1.0, 2.0, 3.0):
        t0 = time.time()
        v = math.sqrt(t)
        PH_t = torch.zeros(J, KZ)
        PH_t[0, 0] = v
        PH_t[1, 0] = v
        PH_t[4, 1] = v
        PH_t[5, 1] = -v
        b_t = torch.full((J,), calibrate_b(mask_ne, PH_t, TARGET_SIZE))
        E_t = energy(mask_ne, b_t, PH_t)
        lt, _ = lift_of(mask_ne, E_t, pairs)

        rng = np.random.default_rng(0)
        p_t = torch.softmax(E_t, 0).numpy()
        draw = rng.choice(mask_ne.shape[0], size=NB, p=p_t)
        uniq, inv = np.unique(draw, return_inverse=True)
        red = mask_ne[torch.as_tensor(uniq)].contiguous()          # distinct baskets only
        cnt = torch.as_tensor(np.bincount(inv).astype(np.float64))
        N = cnt.sum()

        cap = 1.5 * v
        res = {}
        for mode in ("exact", "single", "aniso"):
            bh = torch.full((J,), -1.0, requires_grad=True)
            PH = (torch.randn(J, KZ, generator=torch.Generator().manual_seed(1))
                  * 0.1).requires_grad_(True)
            opt = torch.optim.Adam([bh, PH], lr=0.05)
            gen = torch.Generator().manual_seed(7)
            for step in range(STEPS):
                Ed = energy(red, bh, PH)                            # data term, reduced
                if mode == "exact":
                    lz = torch.logsumexp(energy(mask_ne, bh, PH), 0)
                elif mode == "single":
                    lz = logZ_single(bh, PH, DRAWS, gen)
                else:
                    lz = logZ_aniso(bh, PH, DRAWS, gen)
                ll = ((Ed - lz) * cnt).sum() / N
                opt.zero_grad()
                (-ll).backward()
                opt.step()
                with torch.no_grad():                               # the same norm cap
                    nn_ = PH.norm(dim=1, keepdim=True).clamp_min(1e-12)
                    PH.mul_((cap / nn_).clamp(max=1.0))
            with torch.no_grad():
                lf, szf = lift_of(mask_ne, energy(mask_ne, bh, PH), pairs)
                res[mode] = (float(PH[0] @ PH[1]), lf[(0, 1)], szf)

        print(f"{t:6.2f}{cap**2:7.2f}{lt[(0,1)]:8.3f} | "
              f"{res['exact'][0]:8.3f}{res['single'][0]:8.3f}{res['aniso'][0]:8.3f} | "
              f"{res['exact'][1]:8.3f}{res['single'][1]:8.3f}{res['aniso'][1]:8.3f} | "
              f"{res['exact'][2]:6.2f}{res['single'][2]:6.2f}{res['aniso'][2]:6.2f}"
              f"{time.time()-t0:6.0f}s", flush=True)


if __name__ == "__main__":
    main()
