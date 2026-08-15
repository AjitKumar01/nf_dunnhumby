"""How many draws does log Z need, as a function of ||phi||, at the real catalogue size?

run56 raised the phi cap to 1.20 and aborted at iteration 100: phi reached 0.204 and log Z at
16 draws differed from log Z at 256 by 3.714 nats, so the normaliser guard stopped it.  run55
held phi at 0.218 stably -- but on 256 draws.  So the draw requirement climbs steeply with
phi, and the whole question is how steeply.

The synthetic bench cannot answer this.  It ran 20 products with exact enumeration available;
the quantity that matters here is how badly a 12-dimensional Gaussian proposal covers a
5,455-product catalogue, which only exists at full scale.  s16 tried to probe the coupling
axis and was confounded -- raising lam_max there required filler norms of 4.0, which is not
how the real model reaches high coupling.

So measure it directly.  Take a trained checkpoint, rescale phi to a series of norms, and at
each one compare log Z across draw counts against the highest available reference.  No
training, no gradients: this is a property of the estimator at a fixed parameter setting.

The reference is the largest draw count in the sweep, so the last column is 0 by construction
and the honest reading is the trend, not the absolute number.  A norm whose gap is already
flat by 256 draws is affordable; one still moving at 4096 is not.

Run:  V3_AFFINITY=1 python3 drawcurve.py --ckpt ../../out/v3_run55_best.pt
"""
import argparse
import os
import time

import numpy as np
import torch

from data import build
from features import Features
from fit import Batcher
from ragged import RaggedModel


def log(m):
    print(f"[dc] {m}", flush=True)


def main(a):
    torch.set_default_dtype(torch.float64)
    D = build()
    J, N, C, S = (int(D[k]) for k in ("n_item", "n_user", "n_cat", "n_store"))
    F = Features(J, S, 712)
    Bt = Batcher(D, F, a.nmax)
    m = RaggedModel(J=J, N=N, C=C, K=32, Kz=12, nmax=a.nmax, R=a.R, S=S, Kp=8)
    m.load_state_dict(torch.load(a.ckpt, map_location="cpu"))
    m.double().eval()

    base = m.phi.detach().clone()
    cur = float(base.norm(dim=1).mean())
    log(f"checkpoint {os.path.basename(a.ckpt)}   current mean ||phi|| {cur:.4f}")

    val = np.flatnonzero(D["trip_split"] == 1)[: a.n_trips]
    # log Z materialises [n_draws, n_slots].  At 4096 draws over 96 trips that is 508,771
    # slots x 4096 x 8 bytes = 16.7 GB and the process is killed with no traceback -- which
    # is what the first attempt did.  Trips are independent under log Z, so chunking over
    # them bounds memory at (chunk slots x max draws) regardless of the sweep.
    log(f"{len(val)} validation trips, in chunks of {a.chunk}")

    draws = [int(x) for x in a.draws.split(",")]
    ref = draws[-1]
    log("")
    log(f"gap = log Z(nd) - log Z({ref}), mean over trips, in nats.  "
        f"lam_max is the per-trip max.")
    hdr = "".join(f"{d:>10d}" for d in draws)
    log(f"{'||phi||':>9}{'lam_max':>9}{hdr}{'time':>9}")

    for tgt in [float(x) for x in a.norms.split(",")]:
        t0 = time.time()
        parts = {d: [] for d in draws}
        lams = []
        with torch.no_grad():
            nrm = base.norm(dim=1, keepdim=True).clamp_min(1e-12)
            if a.frac >= 1.0:
                m.phi.copy_(base * (tgt / nrm))      # every product to the same norm
            else:
                # Sparse phi: only the top `frac` of products by current norm carry the
                # target; the rest are pushed to `floor`.  Real complementarity is sparse --
                # most products pair with nothing -- and the uniform sweep is therefore the
                # worst case, not the realistic one.
                k = max(1, int(a.frac * base.shape[0]))
                keep = torch.topk(nrm.flatten(), k).indices
                newp = base * (a.floor / nrm)
                newp[keep] = (base * (tgt / nrm))[keep]
                m.phi.copy_(newp)
            for k in range(0, len(val), a.chunk):
                ix, ctx, lctx, hh, LI, LT, LC, LU = Bt.make(val[k:k + a.chunk])
                m.house, m.ctx = hh, ctx
                for d in draws:
                    g = torch.Generator().manual_seed(11)
                    parts[d].append(m.log_Z(ix, n_draws=d, generator=g, drop_empty=True,
                                            mix_scales=None, aniso=a.aniso).clone())
                try:
                    lams.append(float(m.lambda_max(ix).max()))
                except Exception:
                    pass
        lz = {d: torch.cat(parts[d]) for d in draws}
        lam = max(lams) if lams else float("nan")
        row = "".join(f"{float((lz[ref] - lz[d]).mean()):>10.3f}" for d in draws)
        log(f"{tgt:9.3f}{lam:9.3f}{row}{time.time()-t0:9.0f}s")

    log("")
    log("A norm is affordable when its gap has gone flat by a draw count you can pay for "
        "every iteration.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="../../out/v3_run55_best.pt")
    p.add_argument("--norms", default="0.10,0.20,0.40,0.60,0.96")
    p.add_argument("--draws", default="16,64,256,1024,4096")
    p.add_argument("--n-trips", type=int, default=96)
    p.add_argument("--chunk", type=int, default=4)
    p.add_argument("--frac", type=float, default=1.0)
    p.add_argument("--floor", type=float, default=0.02)
    p.add_argument("--nmax", type=int, default=120)
    p.add_argument("--R", type=int, default=23)
    p.add_argument("--aniso", type=float, default=2.0)
    main(p.parse_args())
