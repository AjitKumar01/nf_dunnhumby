"""Which 109 products kept phi -- and are they the ones that actually co-occur?

run57 gave the model room for a pair lift of 4.2 (max phi'phi = 1.44) and generated a pair
co-occurrence ratio of 0.065 against run39's 0.053.  The capacity was created and barely used.

The top-k mask is the obvious suspect, because of how it selects.  It keeps the products with
the LARGEST CURRENT NORM, so whichever products happen to grow first stay in the mask and keep
growing, and everything else is zeroed every step and can never recover.  That is a
rich-get-richer rule, and nothing in it refers to co-purchase.  If the products it locked onto
are simply the most frequently bought ones, the model spent its entire phi budget on items
that need no complementarity term at all.

Two questions decide it:

  1. Are the retained products the FREQUENT ones, or the co-purchased ones?  Frequency rank
     is the diagnostic -- if the retained set is concentrated at the top of the frequency
     distribution, the mask is tracking popularity, not affinity.

  2. Of the 200 most co-purchased real pairs -- the exact pairs the generation check scores --
     how many have BOTH products retained?  A pair with one product masked to zero has
     phi_j.phi_k = 0 by construction and CANNOT be modelled, however long the run.  That
     number is a hard ceiling on the co-occurrence ratio and is knowable without any fitting.

Run:  V3_AFFINITY=1 python3 whichphi.py --ckpt ../../out/v3_run57_best.pt
"""
import argparse
import os
from collections import Counter

import numpy as np
import torch

from data import build
from ragged import RaggedModel


def log(m):
    print(f"[wp] {m}", flush=True)


def main(a):
    torch.set_default_dtype(torch.float64)
    D = build()
    J, N, C, S = (int(D[k]) for k in ("n_item", "n_user", "n_cat", "n_store"))
    sd = torch.load(a.ckpt, map_location="cpu")
    phi = sd["phi"].double()
    nrm = phi.norm(dim=1).numpy()
    keep = np.flatnonzero(nrm > 1e-12)
    log(f"{os.path.basename(a.ckpt)}: {len(keep)} of {J} products retain phi "
        f"({100*len(keep)/J:.2f}%), norms {nrm[keep].min():.3f}..{nrm[keep].max():.3f}")

    # ---- 1. frequency rank of the retained set ---------------------------------------
    freq = np.bincount(D["line_item"], minlength=J).astype(np.int64)
    order = np.argsort(-freq)                      # rank 0 = most bought
    rank = np.empty(J, np.int64)
    rank[order] = np.arange(J)
    rk = rank[keep]
    log("")
    log(f"frequency rank of retained products (0 = most bought of {J}):")
    log(f"  median {int(np.median(rk))}   mean {rk.mean():.0f}   "
        f"min {rk.min()}   max {rk.max()}")
    for lo, hi in ((0, 100), (100, 500), (500, 2000), (2000, J)):
        n = int(((rk >= lo) & (rk < hi)).sum())
        log(f"  rank {lo:5d}-{hi:5d}: {n:4d} retained  "
            f"({100.0*n/len(keep):5.1f}% of the mask)")

    # ---- 2. coverage of the pairs the evaluation actually scores ----------------------
    ptr = D["line_ptr"]
    li = D["line_item"]
    cnt = Counter()
    ntr = len(ptr) - 1
    for t in range(ntr):
        s, e = int(ptr[t]), int(ptr[t + 1])
        items = np.unique(li[s:e])
        if len(items) < 2 or len(items) > a.max_basket:
            continue
        for x in range(len(items)):
            for y in range(x + 1, len(items)):
                cnt[(int(items[x]), int(items[y]))] += 1
    top = [p for p, _ in cnt.most_common(a.top_pairs)]
    kept = set(int(x) for x in keep)
    both = sum(1 for (x, y) in top if x in kept and y in kept)
    one = sum(1 for (x, y) in top if (x in kept) != (y in kept))
    log("")
    log(f"of the {len(top)} most co-purchased real pairs:")
    log(f"  both products retained : {both:4d}  ({100.0*both/len(top):5.1f}%)  <- modellable")
    log(f"  exactly one retained   : {one:4d}  ({100.0*one/len(top):5.1f}%)")
    log(f"  neither retained       : {len(top)-both-one:4d} "
        f"({100.0*(len(top)-both-one)/len(top):5.1f}%)")
    log("")
    log(f"phi_j.phi_k is identically zero unless BOTH are retained, so {100.0*both/len(top):.1f}% "
        f"is the hard ceiling\non the pair co-occurrence ratio for this mask, before any "
        f"question of fitting.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="../../out/v3_run57_best.pt")
    p.add_argument("--top-pairs", type=int, default=200)
    p.add_argument("--max-basket", type=int, default=40)
    main(p.parse_args())
