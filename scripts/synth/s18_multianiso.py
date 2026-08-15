"""Stretch the proposal along the top-m curvature directions instead of just the top one.

s17 found the anisotropic proposal's worst-recovered pair decays as the planted rank grows --
1.909, 1.780, 1.694, 1.469 at r = 1, 2, 4, 6 against a true lift of 2.191 -- while exact ML
stays flat at 2.1-2.3.  The mechanism is visible in the code: logZ_aniso takes V[:, -1], the
single top eigenvector of Lambda, and widens the proposal along that one axis.  When the truth
has r independent complementarity directions, one stretched axis covers one of them and the
remaining r-1 are sampled from the un-widened Gaussian, so their pairs are under-recovered.

Real grocery has many independent groups -- pasta/sauce shares no direction with
shampoo/conditioner -- so a rank-1 stretch is the wrong shape for the actual problem.

The generalisation is immediate.  Stretching the whole top-m subspace by s means a proposal
covariance of I + (s^2 - 1) V_m V_m', whose log-density costs m*log(s) instead of log(s):

    eps  = e0 + (s-1) (e0 V_m) V_m'
    lq   = -0.5 [ ||perp||^2 + ||r||^2 / s^2 ] - m log s

m = 1 recovers the existing estimator exactly, so this is a strict generalisation and m is the
only thing under test.  The cost is the same 256 draws either way -- only the shape changes,
not the budget, which is what makes this worth testing before anything more elaborate.

Run:  python3 s18_multianiso.py
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
RANK = 6                # the hardest case s17 ran, where aniso fell to 1.469


def build_mask(Jd):
    idx = torch.arange(2 ** Jd, dtype=torch.int64)
    return ((idx.unsqueeze(1) >> torch.arange(Jd, dtype=torch.int64)) & 1).to(torch.float64)


def energy(mask, b, PH):
    v = mask @ PH
    return mask @ b + 0.5 * ((v * v).sum(1) - mask @ (PH ** 2).sum(1))


def erank(PH):
    s = torch.linalg.svdvals(PH)
    return float((s ** 2).sum() ** 2 / (s ** 4).sum().clamp_min(1e-30))


def stats(mask_ne, b, PH, pairs):
    p = torch.softmax(energy(mask_ne, b, PH), 0)
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


def mode_and_V(b, PH, m, steps=6):
    """Laplace mode, plus the top-m eigenvectors of Lambda at that mode."""
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
    return zh, V[:, KZ - m:]                       # [KZ, m], largest eigenvalues last


def logZ_aniso_m(b, PH, nd, gen, m, s=2.0):
    """m = 1 is identical to the existing logZ_aniso."""
    zh, V = mode_and_V(b, PH, m)
    e0 = torch.randn(nd, KZ, generator=gen)
    eps = e0 + (s - 1.0) * (e0 @ V) @ V.T
    r = eps @ V
    perp = eps - r @ V.T
    lq = -0.5 * (perp.pow(2).sum(1) + (r / s).pow(2).sum(1)) - m * math.log(s)
    zs = zh + eps
    w = torch.exp(b.unsqueeze(0) + zs @ PH.T - 0.5 * (PH ** 2).sum(1).unsqueeze(0))
    lf = torch.expm1(torch.log1p(w).sum(1)).clamp_min(1e-300).log()
    lw = (-0.5 * (zs ** 2).sum(1) + lf) - lq
    return torch.logsumexp(lw, 0) - math.log(nd), lw


def main():
    mask = build_mask(J)
    mask_ne = mask[mask.sum(1) > 0].contiguous()
    del mask

    v = math.sqrt(TRUE_T)
    pairs = [(2 * i, 2 * i + 1) for i in range(RANK)]
    PH_t = torch.zeros(J, KZ)
    for i, (a, c) in enumerate(pairs):
        PH_t[a, i] = v
        PH_t[c, i] = v
    b_t = calib_b(mask_ne, PH_t, RANK, SIZE)
    _, size_t, lift_t = stats(mask_ne, b_t, PH_t, pairs)
    print(f"J = {J}, Kz = {KZ}, rank {RANK} truth, each pair at phi'phi = {TRUE_T}, "
          f"mean size {size_t:.2f}")
    print(f"true lift  mean {np.mean(lift_t):.3f}  worst {np.min(lift_t):.3f}"
          f"   (m = 1 is the existing estimator)\n")

    rng = np.random.default_rng(0)
    p_t = torch.softmax(energy(mask_ne, b_t, PH_t), 0).numpy()
    draw = rng.choice(mask_ne.shape[0], size=NB, p=p_t)
    uniq, inv = np.unique(draw, return_inverse=True)
    red = mask_ne[torch.as_tensor(uniq)].contiguous()
    cnt = torch.as_tensor(np.bincount(inv).astype(np.float64))

    cap = 1.5 * v
    print(f"{'m':>3}{'phi.phi':>9}{'L mean':>8}{'L worst':>9}{'erank':>7}{'size':>7}"
          f"{'ESS':>7}{'time':>7}")
    for m in (1, 2, 4, 8, 12):
        t0 = time.time()
        bh = torch.full((J,), -1.0, requires_grad=True)
        PH = (torch.randn(J, KZ, generator=torch.Generator().manual_seed(1))
              * 0.1).requires_grad_(True)
        opt = torch.optim.Adam([bh, PH], lr=0.05)
        gen = torch.Generator().manual_seed(7)
        ess_last = 0.0
        for step in range(STEPS):
            Ed = energy(red, bh, PH)
            lz, lw = logZ_aniso_m(bh, PH, DRAWS, gen, m)
            ll = ((Ed - lz) * cnt).sum() / cnt.sum()
            opt.zero_grad()
            (-ll).backward()
            opt.step()
            with torch.no_grad():
                nn_ = PH.norm(dim=1, keepdim=True).clamp_min(1e-12)
                PH.mul_((cap / nn_).clamp(max=1.0))
                if step == STEPS - 1:
                    ww = torch.softmax(lw.detach(), 0)
                    ess_last = float(1.0 / (ww ** 2).sum() / DRAWS)
        with torch.no_grad():
            _, szf, lff = stats(mask_ne, bh, PH, pairs)
            dots = float(np.mean([float(PH[a] @ PH[c]) for (a, c) in pairs]))
            print(f"{m:3d}{dots:9.3f}{float(np.mean(lff)):8.3f}{float(np.min(lff)):9.3f}"
                  f"{erank(PH.detach()):7.2f}{szf:7.2f}{ess_last:7.3f}"
                  f"{time.time()-t0:6.0f}s", flush=True)

    print(f"\n{'true':>3}{TRUE_T:9.3f}{np.mean(lift_t):8.3f}{np.min(lift_t):9.3f}"
          f"{erank(PH_t):7.2f}{size_t:7.2f}")


if __name__ == "__main__":
    main()
