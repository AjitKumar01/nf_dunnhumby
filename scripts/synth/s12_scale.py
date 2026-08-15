"""Does exact-log-Z recovery survive a bigger catalogue and bigger baskets?

s3 recovered a planted phi almost perfectly -- but at J=12 with a mean basket size of 1.86,
where a basket contains 0.8 pairs on average.  Real baskets hold 6.24 items out of 5,455,
which is 15 pairs.  The pair term therefore carries an order of magnitude more of the energy
in the real regime than in the regime the recovery was demonstrated in.  That gap is large
enough that "recovery works" was never safe to extrapolate from.

So: sweep J in {12, 16, 20} and mean basket size in {2, 6}, holding the planted structure
fixed, with log Z summed exactly over all 2^J subsets every time.  Any degradation here is
the objective or the optimiser meeting scale -- the estimator is not involved at all.

b is calibrated per configuration by bisection so the mean size hits its target; otherwise a
bigger J at fixed b would silently change basket size too, and the two effects could not be
told apart.

Run:  python3 s12_scale.py
"""
import math
import time

import numpy as np
import torch

torch.set_default_dtype(torch.float64)

NB = 40000          # baskets generated per configuration
KFIT = 6            # latent width used by the FIT (truth uses 2 directions)
STEPS = 1200


def build_mask(J):
    """[2^J, J] indicator of every subset, built by bit ops rather than a Python loop."""
    idx = torch.arange(2 ** J, dtype=torch.int64)
    bits = (idx.unsqueeze(1) >> torch.arange(J, dtype=torch.int64)) & 1
    return bits.to(torch.float64)


def energy(mask, b, PH):
    """E(S) = sum_j b_j + sum_{j<k} phi_j.phi_k, vectorised over all subsets."""
    v = mask @ PH
    sq = mask @ (PH ** 2).sum(1)
    return mask @ b + 0.5 * ((v * v).sum(1) - sq)


def stats(mask_ne, E_ne, pairs):
    """Exact P(S) over the non-empty support, then marginals, lifts and mean size."""
    p = torch.softmax(E_ne, 0)
    pi = (mask_ne * p.unsqueeze(1)).sum(0)
    size = float((mask_ne.sum(1) * p).sum())
    lifts = {}
    for (a, c) in pairs:
        joint = float((mask_ne[:, a] * mask_ne[:, c] * p).sum())
        lifts[(a, c)] = joint / max(float(pi[a] * pi[c]), 1e-300)
    return p, pi, size, lifts


def calibrate_b(mask_ne, PH, J, target, lo=-8.0, hi=4.0):
    """Scalar b such that the exact mean non-empty basket size equals `target`."""
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        E = energy(mask_ne, torch.full((J,), mid), PH)
        s = float((mask_ne.sum(1) * torch.softmax(E, 0)).sum())
        if s < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def run(J, target_size, seed=0):
    t0 = time.time()
    mask = build_mask(J)
    ne = mask.sum(1) > 0
    mask_ne = mask[ne].contiguous()
    del mask

    # ---- planted truth: one complementary pair, one substitutable pair ----------------
    PH_t = torch.zeros(J, KFIT)
    PH_t[0, 0] = 1.0
    PH_t[1, 0] = 1.0        # phi_0 . phi_1 = +1.0
    PH_t[4, 1] = 1.0
    PH_t[5, 1] = -1.0       # phi_4 . phi_5 = -1.0
    bval = calibrate_b(mask_ne, PH_t, J, target_size)
    b_t = torch.full((J,), bval)

    pairs = [(0, 1), (4, 5), (0, 2)]
    E_t = energy(mask_ne, b_t, PH_t)
    p_t, pi_t, size_t, lift_t = stats(mask_ne, E_t, pairs)

    # ---- generate baskets by exact multinomial over the full support ------------------
    rng = np.random.default_rng(seed)
    draw = rng.choice(mask_ne.shape[0], size=NB, p=p_t.numpy())
    cnt = torch.as_tensor(np.bincount(draw, minlength=mask_ne.shape[0]).astype(np.float64))

    # ---- fit b and PHI by exact maximum likelihood (log Z summed, never sampled) ------
    bh = torch.full((J,), -1.0, requires_grad=True)
    PH = (torch.randn(J, KFIT, generator=torch.Generator().manual_seed(1))
          * 0.1).requires_grad_(True)
    opt = torch.optim.Adam([bh, PH], lr=0.05)
    N = cnt.sum()
    for step in range(STEPS):
        E = energy(mask_ne, bh, PH)
        ll = ((E - torch.logsumexp(E, 0)) * cnt).sum() / N
        opt.zero_grad()
        (-ll).backward()
        opt.step()

    with torch.no_grad():
        E_f = energy(mask_ne, bh, PH)
        _, _, size_f, lift_f = stats(mask_ne, E_f, pairs)
        d01 = float(PH[0] @ PH[1])
        d45 = float(PH[4] @ PH[5])
        d02 = float(PH[0] @ PH[2])

    print(f"{J:4d}{size_t:9.2f}{size_t*(size_t-1)/2:8.1f} | "
          f"{d01:+8.3f}{d45:+8.3f}{d02:+8.3f} | "
          f"{lift_t[(0,1)]:8.3f}{lift_f[(0,1)]:8.3f} | "
          f"{lift_t[(4,5)]:8.3f}{lift_f[(4,5)]:8.3f} | "
          f"{lift_t[(0,2)]:7.3f}{lift_f[(0,2)]:7.3f} | "
          f"{size_f:7.2f}{time.time()-t0:8.0f}s", flush=True)


if __name__ == "__main__":
    print(f"exact ML recovery, log Z summed over all 2^J subsets, {NB:,} baskets, "
          f"{STEPS} Adam steps")
    print("planted:  phi_0.phi_1 = +1.000   phi_4.phi_5 = -1.000   phi_0.phi_2 = 0 (control)")
    print()
    print(f"{'J':>4}{'size':>9}{'pairs':>8} | {'d01':>8}{'d45':>8}{'d02':>8} | "
          f"{'L01 true':>8}{'fit':>8} | {'L45 true':>8}{'fit':>8} | "
          f"{'L02 t':>7}{'fit':>7} | {'sz fit':>7}{'time':>8}")
    for J in (12, 16, 20):
        for sz in (2.0, 6.0):
            run(J, sz)
