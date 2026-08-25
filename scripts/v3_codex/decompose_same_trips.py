"""Decompose joint set likelihood into size and composition on the shared trips."""
import argparse
import json
import os

import numpy as np
import torch

import baselines as BL
import baselines2 as B2
import evalall as EA
from bench_same_trips import OUT, load_main, paired, strict_load, summarize
from data import build
from features import Features
from fit import Batcher


@torch.no_grad()
def score_main(model, batcher, trips, chunk):
    joint, size, comp, lines = [], [], [], []
    for k in range(0, len(trips), chunk):
        sub = trips[k:k + chunk]
        ix, ctx, lctx, hh, li, lt, lc, _ = batcher.make(sub)
        model.house, model.ctx = hh, ctx
        ll, pn = model.loglik(ix, li, lt, lc, line_ctx=lctx, return_size=True)
        n = torch.bincount(lt, minlength=len(sub))
        lpn = torch.log(pn[torch.arange(len(sub)), n - 1].clamp_min(1e-300))
        joint.extend(ll.tolist()); size.extend(lpn.tolist()); comp.extend((ll - lpn).tolist())
        lines.extend(n.tolist())
    return {"joint": np.asarray(joint), "size": np.asarray(size),
            "composition": np.asarray(comp), "lines": np.asarray(lines)}


@torch.no_grad()
def score_multinomial(model, batcher, trips, R, chunk):
    joint, size, comp, lines = [], [], [], []
    for k in range(0, len(trips), chunk):
        sub = trips[k:k + chunk]
        d = batcher.make(sub)
        ll = model.loglik(d, category_cap=R)
        n = torch.bincount(d["lt"], minlength=len(sub))
        lpn = model.log_pn[n]
        joint.extend(ll.tolist()); size.extend(lpn.tolist()); comp.extend((ll - lpn).tolist())
        lines.extend(n.tolist())
    return {"joint": np.asarray(joint), "size": np.asarray(size),
            "composition": np.asarray(comp), "lines": np.asarray(lines)}


def main(a):
    torch.set_default_dtype(torch.float64)
    D = build()
    J, N, _, S = (int(D[k]) for k in ("n_item", "n_user", "n_cat", "n_store"))
    features = Features(J, S, 712)
    mb = Batcher(D, features, a.nmax)
    bb = BL.Batches(D, features)
    model, _, _ = load_main(os.path.join(OUT, a.main_ckpt), D, 0, a.nmax, a.R)
    law = B2.size_law(D, a.nmax, a.R)
    multi = B2.Multinomial(J, N, S, law, K=32, Kp=8)
    strict_load(multi, os.path.join(OUT, a.multinomial_ckpt), ignore=("log_pn",))
    with torch.no_grad():
        p = torch.as_tensor(law)
        multi.log_pn.copy_(torch.where(p > 0, torch.log(p),
                                       torch.full_like(p, -float("inf"))))
    multi.double().eval()
    result = {"seed": a.seed, "n_trips": a.n_trips, "splits": {}}
    for split in a.splits.split(","):
        trips = EA.sample_split(D, split, a.n_trips, a.nmax, a.R, seed=a.seed)
        print(f"[dec] {split}: main", flush=True)
        ours = score_main(model, mb, trips, a.main_chunk)
        print(f"[dec] {split}: multinomial", flush=True)
        base = score_multinomial(multi, bb, trips, a.R, a.baseline_chunk)
        if not np.array_equal(ours["lines"], base["lines"]):
            raise RuntimeError("line counts differ")
        block = {}
        for part in ("joint", "size", "composition"):
            block[part] = {
                "main": summarize(ours[part], ours["lines"]),
                "multinomial": summarize(base[part], base["lines"]),
                "gap": paired(ours[part], base[part]),
            }
            g = block[part]["gap"]
            print(f"[dec] {split:5s} {part:11s}: main {ours[part].mean():9.4f}  "
                  f"multi {base[part].mean():9.4f}  gap {g['main_minus_baseline']:+.4f} "
                  f"+/- {g['paired_se']:.4f}", flush=True)
        result["splits"][split] = block
    path = os.path.join(OUT, a.output + ".json")
    with open(path, "w") as stream:
        json.dump(result, stream, indent=2)
    print(f"[dec] wrote {path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--main-ckpt", default="v3_run112_blockscaled_safe_best.pt")
    parser.add_argument("--multinomial-ckpt", default="v3_bl_multinom.pt")
    parser.add_argument("--splits", default="valid,test")
    parser.add_argument("--n-trips", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--nmax", type=int, default=120)
    parser.add_argument("--R", type=int, default=23)
    parser.add_argument("--main-chunk", type=int, default=24)
    parser.add_argument("--baseline-chunk", type=int, default=8)
    parser.add_argument("--output", default="v3_same_trips_decomposition")
    main(parser.parse_args())
