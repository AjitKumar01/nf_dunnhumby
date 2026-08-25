"""Paired iteration-400 audit for the repaired basket baselines."""
import argparse
import hashlib
import json
import math
import os
import time

import numpy as np
import torch

from baselines import Batches, Bernoulli, DPP
from baselines2 import NDPP, Shopper
from bench_same_trips import OUT, checkpoint_record, paired, strict_load, summarize
from data import build
from features import Features


def hash_ids(ids):
    return hashlib.sha256(np.ascontiguousarray(ids, dtype=np.int64).tobytes()).hexdigest()


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def model_from_checkpoint(name, path, data, iteration):
    blob = torch.load(path, map_location="cpu", weights_only=False)
    require(blob.get("format") == 3 and blob.get("kind") == "verified-basket-baseline",
            f"{name}: checkpoint has no verified provenance")
    require(blob.get("model_name") == name and int(blob.get("iteration", -1)) == iteration,
            f"{name}: checkpoint is not the requested iteration-{iteration} model")
    cfg, md = blob["config"], blob["data"]
    require(not cfg.get("resume") and int(cfg["R"]) == int(cfg["nmax"]) == 120,
            f"{name}: not a fresh complete-support run")
    require(md.get("affinity") == "1" and int(md["n_item"]) == 5455,
            f"{name}: wrong data universe")
    J, N, S = (int(data[k]) for k in ("n_item", "n_user", "n_store"))
    common = dict(K=cfg["K"], Kp=cfg["Kp"], seed=cfg["seed"],
                  taste_init=cfg["taste_init"])
    if name == "bernoulli":
        model = Bernoulli(J, N, S, **common)
    elif name == "dpp":
        model = DPP(J, N, S, rank=cfg["rank"],
                    interaction_init=cfg["interaction_init"], **common)
    elif name == "ndpp":
        model = NDPP(J, N, S, rank=cfg["rank"], srank=cfg["srank"],
                     interaction_init=cfg["interaction_init"], **common)
    elif name == "shopper":
        model = Shopper(J, N, S, Ki=cfg["interaction_rank"],
                        interaction_init=cfg["interaction_init"], **common)
    else:
        raise ValueError(name)
    strict_load(model, path)
    return model.double().eval(), blob


@torch.no_grad()
def score(model, name, batcher, trips, chunk, shopper_orders, seed):
    values, lines = [], []
    gen = torch.Generator().manual_seed(seed)
    for k in range(0, len(trips), chunk):
        sub = trips[k:k + chunk]
        d = batcher.make(sub)
        if name == "bernoulli":
            ll = model.loglik(d, nmax=120, category_cap=120)
        elif name == "shopper":
            ll = model.loglik(d, n_orders=shopper_orders, gen=gen,
                              exact_max_n=6, max_size=120)
        else:
            ll = model.loglik(d)
        require(len(ll) == len(sub) and bool(torch.isfinite(ll).all()),
                f"{name}: invalid score")
        values.extend(ll.tolist())
        lines.extend(torch.bincount(d["lt"], minlength=len(sub)).tolist())
    return np.asarray(values), np.asarray(lines)


def ndpp_middle(model, dtype):
    k, m = model.rank, 2 * model.srank
    out = torch.zeros(k + m, k + m, dtype=dtype)
    out[:k, :k] = torch.eye(k, dtype=dtype)
    out[k:, k:] = model._C().to(dtype)
    return out


@torch.no_grad()
def dpp_tail_bounds(model, name, batcher, trips, cutoff=120, chunk=8):
    """Chernoff upper bound for P(N>cutoff | N>0) from det(I+zL)."""
    # A coarse log-z grid is sufficient here: accepted bounds must be many orders below
    # the score precision, not merely just below a threshold.
    t_grid = (0.1, 0.2, 0.4, 0.7, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0)
    bounds = []
    for start in range(0, len(trips), chunk):
        d = batcher.make(trips[start:start + chunk])
        q = model.idx(d["item"], d["st"], d["house"], d["ctx"])
        for b in range(d["B"]):
            mask = d["st"] == b
            dg = torch.exp(q[mask].clamp(-12, 6))
            if name == "dpp":
                W = model.V[d["item"][mask]]
                middle = torch.eye(model.rank, dtype=dg.dtype)
            else:
                W = torch.cat([model.V[d["item"][mask]], model.B[d["item"][mask]]], 1)
                middle = ndpp_middle(model, dg.dtype)

            def logdet(z):
                s = 1.0 + z * dg
                A = torch.eye(W.shape[1], dtype=dg.dtype) \
                    + z * middle @ (W / s[:, None]).T @ W
                sign, ld = torch.linalg.slogdet(A)
                require(float(sign) > 0, f"{name}: scaled determinant is not positive")
                return torch.log(s).sum() + ld

            base = logdet(1.0)
            log_nonempty = torch.log1p(-torch.exp((-base).clamp(max=-1e-12)))
            candidates = []
            for t in t_grid:
                candidates.append(float(logdet(math.exp(t)) - base
                                        - t * (cutoff + 1) - log_nonempty))
            bounds.append(min(0.0, min(candidates)))
    return np.asarray(bounds)


def main(args):
    torch.set_default_dtype(torch.float64)
    torch.set_flush_denormal(True)
    require(os.environ.get("V3_AFFINITY") == "1", "set V3_AFFINITY=1")
    data = build()
    valid = np.flatnonzero(data["trip_split"] == 1).astype(np.int64)
    trips = valid[np.random.default_rng(12345).permutation(len(valid))][:384]
    full = np.load(os.path.join(OUT, args.full_per_trip))
    require(np.array_equal(trips, full["trips"]), "stored full score uses another manifest")
    full_joint, lines = full["full_joint"], full["lines"]
    batcher = Batches(data, Features(int(data["n_item"]), int(data["n_store"]), 712))

    result = dict(schema=1, created_unix=time.time(), iteration=args.iteration,
                  manifest=dict(n=384, sha256=hash_ids(trips), ids=trips.tolist()),
                  full=summarize(full_joint, lines), baselines={})
    arrays = dict(trips=trips, lines=lines, full=full_joint)
    for name in ("bernoulli", "dpp", "ndpp", "shopper"):
        suffix = f"_{args.baseline_tag}" if args.baseline_tag else ""
        path = os.path.join(OUT, f"v3_bl_verified_{name}{suffix}.pt")
        model, blob = model_from_checkpoint(name, path, data, args.iteration)
        print(f"[other] scoring {name}", flush=True)
        values, got_lines = score(model, name, batcher, trips,
                                  args.bernoulli_chunk if name == "bernoulli" else args.chunk,
                                  args.shopper_orders, args.seed)
        require(np.array_equal(lines, got_lines), f"{name}: line counts differ")
        record = checkpoint_record(path, blob)
        record["iteration"] = int(blob["iteration"])
        block = dict(score=summarize(values, lines),
                     paired_full_minus_baseline=paired(full_joint, values),
                     checkpoint=record, support={})
        if name in ("dpp", "ndpp"):
            print(f"[other] auditing {name} cardinality tail", flush=True)
            log_bounds = dpp_tail_bounds(model, name, batcher, trips)
            block["support"] = dict(
                law="all nonempty cardinalities; compared to n<=120 via rigorous tail bound",
                max_log_chernoff_bound=float(log_bounds.max()),
                max_chernoff_bound=float(np.exp(log_bounds.max())),
                accepted_numerically_equivalent=bool(log_bounds.max() < math.log(1e-8)))
            arrays[name + "_log_tail_bound"] = log_bounds
        elif name == "shopper":
            block["support"] = dict(law="exact forced checkout at 120",
                                    ordering_samples=args.shopper_orders,
                                    exact_order_sum_through_size=6)
        else:
            block["support"] = dict(law="exact 1<=n<=120 ESP normalizer")
        result["baselines"][name] = block
        arrays[name] = values
        gap = block["paired_full_minus_baseline"]
        print(f"[other] {name:9s} {values.mean():9.4f}; full-gap "
              f"{gap['main_minus_baseline']:+.4f} +/- {gap['paired_se']:.4f}", flush=True)

    stem = os.path.join(OUT, args.output)
    np.savez_compressed(stem + "_per_trip.npz", **arrays)
    with open(stem + ".json", "w") as stream:
        json.dump(result, stream, indent=2)
    print(f"[other] wrote {stem}.json", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--full-per-trip", default="v3_run155_vs_multinomial_matched400_per_trip.npz")
    p.add_argument("--iteration", type=int, default=400)
    p.add_argument("--baseline-tag", default="")
    p.add_argument("--shopper-orders", type=int, default=8192)
    p.add_argument("--seed", type=int, default=20260821)
    p.add_argument("--chunk", type=int, default=8)
    p.add_argument("--bernoulli-chunk", type=int, default=4)
    p.add_argument("--output", default="v3_other_baselines_matched400")
    main(p.parse_args())
