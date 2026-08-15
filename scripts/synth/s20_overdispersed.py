"""Where does size-law recovery fail as the law gets OVERDISPERSED?

s19 recovered four planted size laws essentially exactly -- KL 0.0000 by exact ML, 0.0016-0.0119
by the sampled anisotropic estimator.  But every law it planted was UNDERdispersed, with
variance-to-mean between 0.23 and 0.78, because at J=16 with a near-independent energy the size
law is binomial-like by construction.  Real grocery data sits at 82.7/7.8 = 10.6, an order of
magnitude the other side of 1, so the bench never entered the regime that matters.

Any law can be planted exactly.  P(n) is proportional to A_n exp(-rho_0(n)), where A_n is the
elementary symmetric mass at size n, so setting

    rho_0(n) = log A_n - log P_target(n)

reproduces P_target exactly whatever b and phi are.  The targets here are negative binomials
truncated to 1..J, parameterised by their variance-to-mean ratio.

TWO ROUTES TO OVERDISPERSION, and they stress different machinery:

  rho_0   shapes the size law directly, leaving the items independent given z.  lam_max stays
          small, so the estimator is never stressed.  This asks whether the FITTING can
          represent and recover an overdispersed law at all.

  phi     overdispersion from global positive coupling -- every product loading on a shared
          direction, so trips draw high or low together.  This is how the real model must
          produce it, and it raises lam_max, which is exactly where the sampled estimator was
          measured to break (run58 aborted at lam_max 1.185).

If recovery holds under rho_0 and fails under phi, the real size-law failure is the estimator
meeting the coupling that overdispersion requires -- which would tie the size-law problem and
the co-occurrence problem to one cause.

There is a hard ceiling worth stating: for n in [1, J] with mean m, the largest achievable
variance is (m-1)(J-m), attained by a two-point law at the ends.  At J=20 and mean 5 that is
60, so variance-to-mean cannot exceed 12 however rho_0 is chosen.

Run:  python3 s20_overdispersed.py
"""
import math
import time

import numpy as np
import torch

torch.set_default_dtype(torch.float64)

J = 20
K = 4
NB = 200000
STEPS = 1200
DRAWS = 256
TARGET_MEAN = 5.0


def build_mask(Jd):
    idx = torch.arange(2 ** Jd, dtype=torch.int64)
    return ((idx.unsqueeze(1) >> torch.arange(Jd, dtype=torch.int64)) & 1).to(torch.float64)


def energy(mask, nvec, b, PH, r0):
    v = mask @ PH
    pair = 0.5 * ((v * v).sum(1) - mask @ (PH ** 2).sum(1))
    return mask @ b + pair - r0[nvec]


def log_A(mask, nvec, b, PH, Jd):
    """log of the unnormalised mass at each size, before rho_0 tilts it."""
    E = energy(mask, nvec, b, PH, torch.zeros(Jd + 1, dtype=b.dtype))
    out = torch.full((Jd + 1,), -math.inf, dtype=b.dtype)
    for n in range(Jd + 1):
        sel = nvec == n
        if bool(sel.any()):
            out[n] = torch.logsumexp(E[sel], 0)
    return out


def nb_target(mean, vm, Jd):
    """Negative binomial with the given variance-to-mean, truncated to 1..J."""
    if vm <= 1.0 + 1e-9:                       # Poisson limit
        k = np.arange(1, Jd + 1)
        p = np.exp(k * np.log(mean) - mean - np.array([math.lgamma(x + 1) for x in k]))
    else:
        disp = mean / (vm - 1.0)               # var = mean * (1 + mean/disp)
        k = np.arange(1, Jd + 1)
        pp = disp / (disp + mean)
        p = np.exp([math.lgamma(x + disp) - math.lgamma(disp) - math.lgamma(x + 1)
                    + disp * math.log(pp) + x * math.log(1 - pp) for x in k])
    p = np.clip(p, 1e-12, None)
    return torch.as_tensor(p / p.sum())


def size_law(nvec_ne, E, Jd):
    p = torch.softmax(E, 0)
    law = torch.zeros(Jd + 1, dtype=p.dtype).index_add(0, nvec_ne, p)
    return law[1:] / law[1:].sum()


def stats(law):
    k = torch.arange(1, law.shape[0] + 1, dtype=law.dtype)
    m = float((k * law).sum())
    return m, float((k * k * law).sum() - m * m)


def kl_tv(p_fit, p_true):
    m = p_true > 1e-12
    kl = float((p_true[m] * (p_true[m] / p_fit[m].clamp_min(1e-300)).log()).sum())
    return kl, float(0.5 * (p_fit - p_true).abs().sum())


def lam_max(mask_ne, E, PH):
    p = torch.softmax(E, 0)
    pi = (mask_ne * p.unsqueeze(1)).sum(0)
    w = pi * (1 - pi)
    L = (PH * w.unsqueeze(1)).T @ PH
    L = L + 1e-9 * torch.eye(L.shape[0], dtype=L.dtype)
    return float(torch.linalg.eigvalsh(L)[-1])


def logZ_aniso(b, PH, r0, nd, gen, Jd, s=2.0):
    kz = PH.shape[1]
    z = torch.zeros(1, kz)
    for _ in range(4):
        zz = z.detach().requires_grad_(True)
        w = torch.exp(b + zz @ PH.T - 0.5 * (PH ** 2).sum(1))
        z = torch.autograd.grad(torch.log1p(w).sum(), zz)[0]
    zh = z.detach()
    w0 = torch.exp(b + zh @ PH.T - 0.5 * (PH ** 2).sum(1))[0]
    pi = (w0 / (1 + w0)).clamp(1e-12, 1 - 1e-12)
    L = (PH.detach() * (pi * (1 - pi)).unsqueeze(1)).T @ PH.detach()
    L = L + 1e-9 * torch.eye(L.shape[0], dtype=L.dtype)
    _, V = torch.linalg.eigh(L)
    vt = V[:, -1]
    e0 = torch.randn(nd, kz, generator=gen)
    eps = e0 + (s - 1.0) * (e0 @ vt).unsqueeze(1) * vt.unsqueeze(0)
    r = eps @ vt
    perp = eps - r.unsqueeze(1) * vt.unsqueeze(0)
    lq = -0.5 * (perp.pow(2).sum(1) + (r / s) ** 2) - math.log(s)
    zs = zh + eps
    lw_item = b.unsqueeze(0) + zs @ PH.T - 0.5 * (PH ** 2).sum(1).unsqueeze(0)
    A = torch.zeros(nd, Jd + 1, dtype=b.dtype)
    A[:, 0] = 1.0
    for j in range(Jd):
        A = A + torch.nn.functional.pad(A[:, :-1], (1, 0)) * torch.exp(lw_item[:, j:j + 1])
    lf = torch.logsumexp(A.clamp_min(1e-300).log() - r0.unsqueeze(0), dim=1)
    return torch.logsumexp((-0.5 * (zs ** 2).sum(1) + lf) - lq, 0) - math.log(nd)


def run_one(mask_ne, nvec_ne, b_t, PH_t, r0_t, label, modes=("free", "aniso")):
    E_t = energy(mask_ne, nvec_ne, b_t, PH_t, r0_t)
    law_t = size_law(nvec_ne, E_t, J)
    mt, vt = stats(law_t)
    lm = lam_max(mask_ne, E_t, PH_t)
    print(f"{label:>24} | {'TRUTH':>6} | {mt:6.2f}{vt:7.2f}{vt/mt:6.2f}{lm:8.3f} | "
          f"{0.0:8.4f}{0.0:7.3f}", flush=True)

    p_t = torch.softmax(E_t, 0).numpy()
    draw = np.random.default_rng(0).choice(mask_ne.shape[0], size=NB, p=p_t)
    uniq, inv = np.unique(draw, return_inverse=True)
    red = mask_ne[torch.as_tensor(uniq)].contiguous()
    redn = nvec_ne[torch.as_tensor(uniq)]
    cnt = torch.as_tensor(np.bincount(inv).astype(np.float64))

    for mode in modes:
        t0 = time.time()
        bh = torch.full((J,), -0.5, requires_grad=True)
        PH = (torch.randn(J, K, generator=torch.Generator().manual_seed(1))
              * 0.1).requires_grad_(True)
        r0 = torch.zeros(J + 1, requires_grad=True)
        opt = torch.optim.Adam([bh, PH, r0], lr=0.05)
        gen = torch.Generator().manual_seed(7)
        for _ in range(STEPS):
            Ed = energy(red, redn, bh, PH, r0)
            lz = (logZ_aniso(bh, PH, r0, DRAWS, gen, J) if mode == "aniso"
                  else torch.logsumexp(energy(mask_ne, nvec_ne, bh, PH, r0), 0))
            ll = ((Ed - lz) * cnt).sum() / cnt.sum()
            opt.zero_grad()
            (-ll).backward()
            opt.step()
        with torch.no_grad():
            Ef = energy(mask_ne, nvec_ne, bh, PH, r0)
            law_f = size_law(nvec_ne, Ef, J)
            mf, vf = stats(law_f)
            kl, tv = kl_tv(law_f, law_t)
            lf_lam = lam_max(mask_ne, Ef, PH.detach())
        print(f"{'':>24} | {mode:>6} | {mf:6.2f}{vf:7.2f}{vf/mf:6.2f}{lf_lam:8.3f} | "
              f"{kl:8.4f}{tv:7.3f}{time.time()-t0:6.0f}s", flush=True)


def main():
    mask = build_mask(J)
    nvec = mask.sum(1).long()
    ne = nvec > 0
    mask_ne, nvec_ne = mask[ne].contiguous(), nvec[ne]
    del mask
    vmax = (TARGET_MEAN - 1) * (J - TARGET_MEAN)
    print(f"J = {J}, target mean {TARGET_MEAN}, {NB:,} baskets, {STEPS} steps")
    print(f"variance ceiling for n in [1,{J}] at mean {TARGET_MEAN} is "
          f"(m-1)(J-m) = {vmax:.0f}, so v/m cannot exceed {vmax/TARGET_MEAN:.1f}")
    print(f"real grocery sits at 82.7/7.8 = 10.6\n")
    print(f"{'planted':>24} | {'fit':>6} | {'mean':>6}{'var':>7}{'v/m':>6}{'lam':>8} | "
          f"{'KL':>8}{'TV':>7}{'time':>7}")

    import sys
    route = sys.argv[1] if len(sys.argv) > 1 else "both"
    # ---- route 1: overdispersion shaped by rho_0, items weakly coupled -------------
    b_t = torch.full((J,), -1.0)
    PH_t = torch.zeros(J, K)
    PH_t[0, 0] = 1.0
    PH_t[1, 0] = 1.0
    lA = log_A(mask_ne, nvec_ne, b_t, PH_t, J)
    for vm in (() if route == 'phi' else (1.0, 2.0, 4.0, 7.0, 10.0)):
        tgt = nb_target(TARGET_MEAN, vm, J)
        r0_t = torch.zeros(J + 1)
        r0_t[1:] = lA[1:] - tgt.log()
        r0_t = r0_t - r0_t[1]
        run_one(mask_ne, nvec_ne, b_t, PH_t, r0_t, f"rho_0 route  v/m={vm:.0f}")
    print()

    # ---- route 2: overdispersion from global coupling, which is how the real model does it
    for w in (0.3, 0.6, 0.9):
        g = torch.Generator().manual_seed(4)
        PH_c = torch.zeros(J, K)
        PH_c[:, 0] = w                          # every product on ONE shared direction
        PH_c[0, 1] = 1.0
        PH_c[1, 1] = 1.0                        # plus the usual planted pair
        run_one(mask_ne, nvec_ne, torch.full((J,), -1.0), PH_c,
                torch.zeros(J + 1), f"phi route    w={w:.1f}")


if __name__ == "__main__":
    main()
