"""Is positive n^2 curvature in rho_0 what stops the pair term saturating?

s20 showed that global positive coupling with rho_0 = 0 saturates: at w = 0.6 every basket
contained all 20 products (mean 19.94, variance 0.06).  The mechanism is that the pair term
sum_{j<k} phi_j.phi_k grows as n(n-1)/2 while the linear term grows as n, so past some size
adding another product is always favourable.

rho_0 is the only term that can grow quadratically the other way.  If that is what holds the
size law together, then:

  1. a rho_0 planted to produce a sensible size law under strong coupling must carry positive
     second difference, and it should be close to w^2 -- the pair term's own curvature;
  2. restricting rho_0 to be LINEAR in n must bring the saturation back, however well the
     linear part is fitted;
  3. a fully free rho_0 must recover.

Point 2 is the real test.  A free rho_0 recovering proves only that the model is flexible
enough; showing that removing exactly the quadratic freedom breaks it is what identifies the
quadratic as the mechanism.

This matters because project_var adds c * n^2 to rho_0 with c clamped NON-NEGATIVE, which is
exactly this counterweight.  It was removed from the fit two runs ago on the grounds that the
clamp made it a one-way ratchet -- and run62 (with it) has 0.8% of trips at the size ceiling
against run63's 2.3% (without).

Run:  python3 s21_counterweight.py
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


def build_mask(Jd):
    idx = torch.arange(2 ** Jd, dtype=torch.int64)
    return ((idx.unsqueeze(1) >> torch.arange(Jd, dtype=torch.int64)) & 1).to(torch.float64)


def energy(mask, nvec, b, PH, r0):
    v = mask @ PH
    pair = 0.5 * ((v * v).sum(1) - mask @ (PH ** 2).sum(1))
    return mask @ b + pair - r0[nvec]


def log_A(mask, nvec, b, PH, Jd):
    E = energy(mask, nvec, b, PH, torch.zeros(Jd + 1, dtype=b.dtype))
    out = torch.full((Jd + 1,), -math.inf, dtype=b.dtype)
    for n in range(Jd + 1):
        sel = nvec == n
        if bool(sel.any()):
            out[n] = torch.logsumexp(E[sel], 0)
    return out


def nb_target(mean, vm, Jd):
    disp = mean / max(vm - 1.0, 1e-6)
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


def main():
    mask = build_mask(J)
    nvec = mask.sum(1).long()
    ne = nvec > 0
    mask_ne, nvec_ne = mask[ne].contiguous(), nvec[ne]
    del mask
    nn = torch.arange(J + 1, dtype=torch.float64)

    print(f"J = {J}, {NB:,} baskets, {STEPS} steps.  Coupling w puts every product on one "
          f"shared direction,\nso the pair term contributes 0.5 * w^2 * n(n-1) and its second "
          f"difference in n is exactly w^2.\n")

    for w in (0.0, 0.2, 0.3, 0.4):
        PH_t = torch.zeros(J, K)
        PH_t[:, 0] = w
        PH_t[0, 1] = 1.0
        PH_t[1, 1] = 1.0                      # the usual planted pair, on its own direction
        b_t = torch.full((J,), -1.0)
        lA = log_A(mask_ne, nvec_ne, b_t, PH_t, J)
        tgt = nb_target(5.0, 2.0, J)          # the size law we want to hold under coupling
        r0_t = torch.zeros(J + 1)
        r0_t[1:] = lA[1:] - tgt.log()
        r0_t = r0_t - r0_t[1]

        # claim 1: the planted rho_0's curvature should track w^2
        d2 = (r0_t[3:J + 1] - 2 * r0_t[2:J] + r0_t[1:J - 1])
        E_t = energy(mask_ne, nvec_ne, b_t, PH_t, r0_t)
        law_t = size_law(nvec_ne, E_t, J)
        mt, vt = stats(law_t)
        print(f"w = {w:.1f}   planted size law mean {mt:5.2f} var {vt:6.2f} "
              f"(v/m {vt/mt:.2f})   mean rho_0'' {float(d2.mean()):+.4f}  vs w^2 {w*w:.4f}")

        p_t = torch.softmax(E_t, 0).numpy()
        draw = np.random.default_rng(0).choice(mask_ne.shape[0], size=NB, p=p_t)
        uniq, inv = np.unique(draw, return_inverse=True)
        red = mask_ne[torch.as_tensor(uniq)].contiguous()
        redn = nvec_ne[torch.as_tensor(uniq)]
        cnt = torch.as_tensor(np.bincount(inv).astype(np.float64))

        for mode in ("rho0 free", "rho0 linear"):
            t0 = time.time()
            bh = torch.full((J,), -0.5, requires_grad=True)
            PH = (torch.randn(J, K, generator=torch.Generator().manual_seed(1))
                  * 0.1).requires_grad_(True)
            if mode == "rho0 free":
                r0v = torch.zeros(J + 1, requires_grad=True)
                params = [bh, PH, r0v]
                get_r0 = lambda: r0v
            else:
                # rho_0 = a + b*n only: every quadratic degree of freedom removed, nothing
                # else changed.  This is the test that identifies the quadratic term.
                ab = torch.zeros(2, requires_grad=True)
                params = [bh, PH, ab]
                get_r0 = lambda: ab[0] + ab[1] * nn
            opt = torch.optim.Adam(params, lr=0.05)
            for _ in range(STEPS):
                r0c = get_r0()
                Ed = energy(red, redn, bh, PH, r0c)
                lz = torch.logsumexp(energy(mask_ne, nvec_ne, bh, PH, r0c), 0)
                ll = ((Ed - lz) * cnt).sum() / cnt.sum()
                opt.zero_grad()
                (-ll).backward()
                opt.step()
            with torch.no_grad():
                law_f = size_law(nvec_ne, energy(mask_ne, nvec_ne, bh, PH, get_r0()), J)
                mf, vf = stats(law_f)
                kl, tv = kl_tv(law_f, law_t)
                cap = float(law_f[int(0.85 * J):].sum())
            print(f"        {mode:>12} | mean {mf:6.2f} var {vf:7.2f} "
                  f"(v/m {vf/max(mf,1e-9):5.2f}) | KL {kl:8.4f} TV {tv:6.3f} "
                  f"| P(n>{int(0.85*J)}) {100*cap:5.1f}% | {time.time()-t0:4.0f}s", flush=True)
        print()


if __name__ == "__main__":
    main()
