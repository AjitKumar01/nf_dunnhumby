"""How much lift does a given phi'phi buy, as a function of how RARE the pair is?

s12 showed the same planted phi_0.phi_1 = +1.000 produces very different lift depending on
regime -- 1.53 at J=12/size 2, but only 1.13 at J=12/size 6.  Both moves act through crowding:
when products are common they compete for room in the basket, and part of the pair bonus is
spent overcoming that competition rather than showing up as lift.

The first version of this script tried to reach the dilute regime by shrinking mean basket
size below 1.  That is impossible: the model conditions on non-empty baskets, so mean size has
a hard floor at 1.0, and as it approaches that floor every basket is a singleton and the pair
can never co-occur at all (lift 0.000 -- which is what that run printed).

Dilution has to be created the other way: make the PAIR rare while other products carry the
basket size.  Products 0, 1, 2 get their own b, swept down until their marginals reach the
real data's pi = 6.24/5455 = 0.00114; products 3.. carry the rest, with their b calibrated so
total mean size stays at 6.  Product 2 is the neutral control and shares the pair's b, so the
comparison is like-for-like at every dilution.

This is the number the whole project turns on.  Grocery co-occurrence needs a lift near 2.5.
If lift -> exp(phi'phi) as the pair gets rare, then the real fit needs phi'phi ~ ln(2.5) =
0.92 -- and s13 found the anisotropic estimator still works at phi'phi = 1 but fails at 2.
Whether 0.92 sits inside or outside the working range is the difference between this being
fixable and being a ceiling.

Run:  python3 s14_dilution.py
"""
import math
import time

import numpy as np
import torch

torch.set_default_dtype(torch.float64)

J = 20
K = 4
PAIR = (0, 1)
CTRL = (0, 2)
SIZE = 6.0


def build_mask(Jd):
    idx = torch.arange(2 ** Jd, dtype=torch.int64)
    return ((idx.unsqueeze(1) >> torch.arange(Jd, dtype=torch.int64)) & 1).to(torch.float64)


def measure(mask_ne, b, PH):
    """Exact marginals, mean size and pair lifts under P(S) over non-empty subsets."""
    v = mask_ne @ PH
    E = mask_ne @ b + 0.5 * ((v * v).sum(1) - mask_ne @ (PH ** 2).sum(1))
    p = torch.softmax(E, 0)
    pi = (mask_ne * p.unsqueeze(1)).sum(0)
    size = float((mask_ne.sum(1) * p).sum())
    out = {}
    for (a, c) in (PAIR, CTRL):
        joint = float((mask_ne[:, a] * mask_ne[:, c] * p).sum())
        out[(a, c)] = joint / max(float(pi[a] * pi[c]), 1e-300)
    return pi, size, out


def fill_for_size(mask_ne, b_pair, PH, target):
    """Given the pair's b, find the filler b so total mean size hits `target`."""
    lo, hi = -14.0, 8.0
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        b = torch.full((J,), mid)
        b[0] = b[1] = b[2] = b_pair
        _, s, _ = measure(mask_ne, b, PH)
        if s < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def main():
    mask = build_mask(J)
    mask_ne = mask[mask.sum(1) > 0].contiguous()
    del mask
    print(f"J = {J}, total mean basket size held at {SIZE}; products 0,1,2 swept down in "
          f"rarity,\nproducts 3..{J-1} carry the size.  (0,1) is the planted pair, "
          f"(0,2) the neutral control.")
    print(f"real data sits at pi = 6.24 / 5455 = 0.00114\n")

    for t in (0.92, 2.0):
        v = math.sqrt(t)
        PH = torch.zeros(J, K)
        PH[0, 0] = v
        PH[1, 0] = v
        print(f"planted phi'phi = {t:+.3f}    dilute limit exp(phi'phi) = {math.exp(t):.3f}")
        print(f"{'b_pair':>8}{'pi_pair':>10}{'size':>7}{'lift(0,1)':>11}{'control':>9}"
              f"{'% of exp':>10}{'time':>7}")
        for b_pair in (-1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0):
            t0 = time.time()
            bf = fill_for_size(mask_ne, b_pair, PH, SIZE)
            b = torch.full((J,), bf)
            b[0] = b[1] = b[2] = b_pair
            pi, s, lf = measure(mask_ne, b, PH)
            print(f"{b_pair:8.1f}{float(pi[0]):10.5f}{s:7.2f}{lf[PAIR]:11.3f}"
                  f"{lf[CTRL]:9.3f}{100*lf[PAIR]/math.exp(t):9.1f}%"
                  f"{time.time()-t0:6.0f}s", flush=True)
        print()


if __name__ == "__main__":
    main()
