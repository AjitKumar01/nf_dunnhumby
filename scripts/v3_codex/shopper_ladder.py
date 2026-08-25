"""Reproducible ordering ladder for SHOPPER on the shared evaluation manifest."""
import argparse
import json
import os
import time

import numpy as np
import torch

import evalall as EA
from baselines import Batches
from baselines2 import Shopper
from bench_same_trips import baseline_scores, strict_load
from data import build
from features import Features


OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "out"))


def main(a):
    torch.set_default_dtype(torch.float64)
    D = build()
    J, N, S = (int(D[k]) for k in ("n_item", "n_user", "n_store"))
    batcher = Batches(D, Features(J, S, 712))
    model = Shopper(J, N, S, K=32, Kp=8)
    ckpt = os.path.join(OUT, a.ckpt)
    strict_load(model, ckpt)
    model.double().eval()
    result = dict(checkpoint=a.ckpt, n_trips=a.n_trips, trip_seed=a.seed,
                  exact_max_n=a.exact_max_n, splits={})
    for si, split in enumerate(a.splits.split(",")):
        trips = EA.sample_split(D, split, a.n_trips, a.nmax, a.R, seed=a.seed)
        block = {}
        for orders in a.ladder:
            estimates = []
            started = time.time()
            for rep in range(a.reps):
                scores, _ = baseline_scores(
                    model, batcher, trips, "shopper", a.chunk,
                    seed=a.seed + 1009 * si + 1000003 * rep,
                    shopper_orders=orders, exact_max_n=a.exact_max_n)
                estimates.append(float(scores.mean()))
            x = np.asarray(estimates)
            block[str(orders)] = dict(mean=float(x.mean()),
                                      replicate_se=(float(x.std(ddof=1) / np.sqrt(len(x)))
                                                    if len(x) > 1 else None),
                                      replicates=estimates,
                                      minutes=(time.time() - started) / 60)
            print(f"[lad] {split:5s} {orders:5d}: {x.mean():9.4f}  "
                  f"rep-SE {block[str(orders)]['replicate_se']}  "
                  f"{block[str(orders)]['minutes']:.2f} min", flush=True)
        result["splits"][split] = block
    path = os.path.join(OUT, a.output + ".json")
    with open(path, "w") as stream:
        json.dump(result, stream, indent=2)
    print(f"[lad] wrote {path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="v3_bl_shopper.pt")
    parser.add_argument("--splits", default="valid,test")
    parser.add_argument("--n-trips", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--nmax", type=int, default=120)
    parser.add_argument("--R", type=int, default=23)
    parser.add_argument("--chunk", type=int, default=8)
    parser.add_argument("--exact-max-n", type=int, default=6)
    parser.add_argument("--ladder", type=int, nargs="+", default=[128, 512, 2048])
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--output", default="v3_shopper_ladder_verified")
    main(parser.parse_args())
