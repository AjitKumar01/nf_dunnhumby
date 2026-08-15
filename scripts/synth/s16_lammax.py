"""Does the dilute-regime recovery survive REALISTIC aggregate coupling?

s15 recovered the planted pair almost exactly at the strength grocery needs (lift 2.214 vs a
true 2.191) -- but only two of the twenty products carried any phi at all.  In the real model
all 5,455 carry phi, and what the estimator has to integrate over is not one pair but their
sum:

    Lambda = sum_j pi_j (1 - pi_j) phi_j phi_j'      lam_max = its top eigenvalue

Because sum_j pi_j IS the mean basket size, lam_max scales roughly as (basket size) x ||phi||^2
in the bench and in the real data alike -- J drops out.  So J=20 with mean size 6 can reproduce
the real fit's coupling faithfully, which is the one thing this bench can do cheaply.

s15 sat at lam_max ~ 0.09.  run55 ran at 3.610.  That is a 40x gap, and lam_max is precisely
the quantity the norm caps were imposed to control, so it is the gap that decides whether
raising the cap fixes the fit or just reproduces the saturation.

Design: products 0,1,2 stay rare (b = -3, pi ~ 0.05) with the planted phi'phi = 0.92 on (0,1);
products 3.. are filler carrying the basket size, and are given random phi directions whose
norm is bisected until lam_max hits each target.  b is recalibrated inside that search so mean
size stays at 6 -- otherwise raising the filler norm would change size and coupling together.

Run:  python3 s16_lammax.py
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


def report(mask_ne, b, PH):
    E = energy(mask_ne, b, PH)
    p = torch.softmax(E, 0)
    pi = (mask_ne * p.unsqueeze(1)).sum(0)
    size = float((mask_ne.sum(1) * p).sum())
    lift = {}
    for (a, c) in ((0, 1), (0, 2)):
        joint = float((mask_ne[:, a] * mask_ne[:, c] * p).sum())
        lift[(a, c)] = joint / max(float(pi[a] * pi[c]), 1e-300)
    return pi, size, lift


def lam_max(pi, PH):
    w = pi * (1.0 - pi)
    L = (PH * w.unsqueeze(1)).T @ PH
    return float(torch.linalg.eigvalsh(L)[-1])


def calib_b(mask_ne, PH, target, iters=34):
    lo, hi = -14.0, 8.0
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        b = torch.full((J,), mid)
        b[0] = b[1] = b[2] = B_PAIR
        _, s, _ = report(mask_ne, b, PH)
        if s < target:
            lo = mid
        else:
            hi = mid
    b = torch.full((J,), 0.5 * (lo + hi))
    b[0] = b[1] = b[2] = B_PAIR
    return b


def make_truth(mask_ne, lam_target, seed=3):
    """Planted pair fixed; filler norm bisected until lam_max hits its target."""
    v = math.sqrt(TRUE_T)
    g = torch.Generator().manual_seed(seed)
    dirs = torch.randn(J, KZ, generator=g)
    dirs = dirs / dirs.norm(dim=1, keepdim=True)
    lo, hi = 0.0, 4.0
    b = PH = None
    for _ in range(22):
        w = 0.5 * (lo + hi)
        PH = dirs * w
        PH[0] = 0.0
        PH[1] = 0.0
        PH[2] = 0.0
        PH[0, 0] = v
        PH[1, 0] = v                      # planted pair, untouched by the sweep
        b = calib_b(mask_ne, PH, SIZE)
        pi, _, _ = report(mask_ne, b, PH)
        if lam_max(pi, PH) < lam_target:
            lo = w
        else:
            hi = w
    return b, PH, 0.5 * (lo + hi)


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
    print(f"J = {J}, Kz = {KZ}, mean size {SIZE}, planted phi'phi = {TRUE_T} (= ln 2.5) on a "
          f"rare pair (pi ~ 0.05).\nFiller products carry phi at random directions, scaled "
          f"until lam_max hits each target.\nrun55 ran at lam_max = 3.610.\n")
    print(f"{'lam tgt':>8}{'lam act':>9}{'w fill':>8}{'L true':>8} | "
          f"{'exact':>8}{'single':>8}{'aniso':>8} | {'L exact':>8}{'L sgl':>8}{'L ani':>8} | "
          f"{'sz ex':>6}{'sz sg':>6}{'sz an':>6}{'time':>7}")

    for lam_t in (0.5, 1.5, 3.0, 5.0):
        t0 = time.time()
        b_t, PH_t, wfill = make_truth(mask_ne, lam_t)
        pi_t, size_t, lift_t = report(mask_ne, b_t, PH_t)
        lam_a = lam_max(pi_t, PH_t)

        rng = np.random.default_rng(0)
        p_t = torch.softmax(energy(mask_ne, b_t, PH_t), 0).numpy()
        draw = rng.choice(mask_ne.shape[0], size=NB, p=p_t)
        uniq, inv = np.unique(draw, return_inverse=True)
        red = mask_ne[torch.as_tensor(uniq)].contiguous()
        cnt = torch.as_tensor(np.bincount(inv).astype(np.float64))

        cap = 1.5 * max(math.sqrt(TRUE_T), wfill)
        res = {}
        for mode in ("exact", "single", "aniso"):
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
                _, szf, lff = report(mask_ne, bh, PH)
                res[mode] = (float(PH[0] @ PH[1]), lff[(0, 1)], szf)

        print(f"{lam_t:8.2f}{lam_a:9.3f}{wfill:8.3f}{lift_t[(0,1)]:8.3f} | "
              f"{res['exact'][0]:8.3f}{res['single'][0]:8.3f}{res['aniso'][0]:8.3f} | "
              f"{res['exact'][1]:8.3f}{res['single'][1]:8.3f}{res['aniso'][1]:8.3f} | "
              f"{res['exact'][2]:6.2f}{res['single'][2]:6.2f}{res['aniso'][2]:6.2f}"
              f"{time.time()-t0:6.0f}s", flush=True)


if __name__ == "__main__":
    main()
