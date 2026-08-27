"""Build a spectral phi initialisation from the empirical co-purchase log-lift.

phi = 0 is a SADDLE: dE(S)/dphi_j = sum_{k in S} phi_k vanishes there, so the gradient is
proportional to phi and escape from a small seed is exponential.  Measured: phi went
0.03 -> 0.099 over 2,600 updates while the data implies ~0.93.

But the model's own first-order relation gives phi in closed form.  For weak interactions

    log [ P(j,k in S) / (P(j in S) P(k in S)) ]  ~  phi_j' phi_k,

so phi is the rank-Kz factorisation of the empirical log-lift matrix -- solvable in one
shot, no escape dynamics.  Measured on the converged 30-product model, this beats the
SGD-trained phi by +0.022 nats on held-out data.

Writes basket_input/v3_phiinit_<mask>_r<Kz>.npy, loadable with --phi-init-file.
"""
import argparse, itertools, math, os
from collections import defaultdict
import numpy as np
import torch
from data import build


def main(a):
    D = build()
    J = int(D["n_item"])
    lp, li = D["line_ptr"], D["line_item"]
    tr = np.flatnonzero(D["trip_split"] == 0)          # TRAINING trips only
    cnt = np.zeros(J); pair = defaultdict(float); nb = 0
    for t in tr:
        it = np.unique(li[int(lp[t]):int(lp[t + 1])])
        if not (2 <= len(it) <= a.max_basket):
            continue
        nb += 1; cnt[it] += 1
        for x, y in itertools.combinations(it.tolist(), 2):
            pair[(x, y)] += 1.0
    p = cnt / max(nb, 1)
    mask = np.load(a.mask) if a.mask else np.ones(J)
    keep = np.flatnonzero(mask)
    pos = -np.ones(J, dtype=np.int64); pos[keep] = np.arange(len(keep))
    M = np.zeros((len(keep), len(keep)))
    used = 0
    for (x, y), c in pair.items():
        ix_, iy = pos[x], pos[y]
        if ix_ < 0 or iy < 0:
            continue
        e = p[x] * p[y] * nb
        if e < 1.0 or c < a.min_pair:
            continue
        M[ix_, iy] = M[iy, ix_] = math.log(c / e); used += 1
    ev, U = torch.linalg.eigh(torch.as_tensor(M))
    idx = torch.argsort(ev, descending=True)[: a.rank]
    F = U[:, idx] * ev[idx].clamp_min(0.0).sqrt()      # positive eigenpairs = attraction
    F = F * a.scale
    nrm = F.norm(dim=1, keepdim=True)
    F = F * (a.phi_max / nrm.clamp_min(1e-12)).clamp(max=1.0)
    out = np.zeros((J, a.rank))
    out[keep] = np.asarray(F)
    tag = os.path.splitext(os.path.basename(a.mask))[0] if a.mask else "all"
    dst = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                       "basket_input", f"v3_phiinit_{tag}_r{a.rank}.npy")
    np.save(dst, out)
    print(f"[spec] {nb:,} training baskets, {used:,} usable pairs over {len(keep)} products")
    print(f"[spec] rank {a.rank} explains "
          f"{100*float((ev[idx]**2).sum()/(ev**2).sum()):.1f}% of the log-lift spectrum")
    print(f"[spec] |phi_j|: median {float(F.norm(dim=1).median()):.3f}, "
          f"max {float(F.norm(dim=1).max()):.3f}")
    print(f"[spec] wrote {os.path.normpath(dst)}")


if __name__ == "__main__":
    q = argparse.ArgumentParser()
    q.add_argument("--mask", default="")
    q.add_argument("--rank", type=int, default=4)
    q.add_argument("--scale", type=float, default=0.4)
    q.add_argument("--phi-max", type=float, default=0.6)
    q.add_argument("--min-pair", type=int, default=5)
    q.add_argument("--max-basket", type=int, default=40)
    main(q.parse_args())
