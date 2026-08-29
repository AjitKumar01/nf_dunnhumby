#!/usr/bin/env python3
"""Deterministic version-4 marginal-incidence MRR for fixed-rank checkpoints."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

os.environ.setdefault("V3_AFFINITY", "1")

import numpy as np
import torch

from audit_particle_counterfactual_generation import ROOT, load_checkpoint
from data import build
from eval_mrr_cutoffs import popularity_ranks
from features import Features
from fit import Batcher, rec_eval
from ragged import smolyak_grid


torch.set_default_dtype(torch.float64)


def metrics(ranks):
    ranks = np.asarray(ranks, dtype=float)
    reciprocal = 1.0 / ranks
    result = {
        "cases": int(len(ranks)),
        "mrr": float(reciprocal.mean()),
        "mrr_se": float(reciprocal.std(ddof=1) / math.sqrt(len(ranks))),
        "median_rank": float(np.median(ranks)),
    }
    for cutoff in (5, 10, 20, 100):
        hit = ranks <= cutoff
        result[f"mrr_at_{cutoff}"] = float(np.where(hit, reciprocal, 0.0).mean())
        result[f"recall_at_{cutoff}"] = float(hit.mean())
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--trips", type=int, default=2000)
    parser.add_argument("--chunk", type=int, default=24)
    parser.add_argument("--rank", type=int, default=0,
                        help="expected active rank; <=0 infers it from the checkpoint")
    parser.add_argument("--level", type=int, default=0,
                        help="Smolyak level; <=0 uses active_rank+2")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--conditioned", action="store_true",
                        help="score version-4 basket completion conditional on revealed items")
    parser.add_argument("--output", type=Path,
                        default=Path("out/v3_smolyak_rank8_mrr.json"))
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    data = build()
    ckpt = args.ckpt if args.ckpt.is_absolute() else ROOT / args.ckpt
    model, blob, meta = load_checkpoint(ckpt, data)
    singular = torch.linalg.svdvals(model.phi)
    active_rank = int((singular > singular[0] * 1e-10).sum())
    if args.rank > 0 and active_rank != args.rank:
        raise RuntimeError(
            f"checkpoint active interaction rank is {active_rank}, not {args.rank}")
    level = args.level if args.level > 0 else active_rank + 2
    active_nodes, weights = smolyak_grid(active_rank, level)
    nodes = torch.zeros(len(weights), model.Kz, dtype=model.phi.dtype)
    nodes[:, :active_rank] = active_nodes
    model.quad = (nodes, weights)
    model.quad_a = None
    model.eval()
    batcher = Batcher(data, Features(int(data["n_item"]), int(data["n_store"]), 712),
                      int(meta["nmax"]))
    split = {"validation": 1, "test": 2}[args.split]
    population = np.flatnonzero((data["trip_split"] == split) &
                                (data["trip_nlines"] <= int(meta["nmax"])))
    trips = population[np.random.default_rng(12345).permutation(len(population))[:args.trips]]
    ranks = rec_eval(model, batcher, trips, seed=0, chunk=args.chunk,
                     return_ranks=True, conditioned=args.conditioned)
    pop = popularity_ranks(data, trips, seed=0)
    if len(pop) != len(ranks):
        raise RuntimeError("model and popularity retained different holdout cases")
    reciprocal_gain = 1.0 / ranks - 1.0 / pop
    gain_se = float(reciprocal_gain.std(ddof=1) / math.sqrt(len(ranks)))
    output = {
        "checkpoint": str(ckpt),
        "checkpoint_iteration": int(blob["iter"]),
        "active_rank": active_rank,
        "smolyak_level": level,
        "smolyak_nodes": len(weights),
        "protocol": (
            "deterministic conditional basket-completion incidence; complete support"
            if args.conditioned else
            "deterministic unconditional marginal incidence pi=dlogZ/db; complete support"),
        "model": metrics(ranks),
        "popularity": metrics(pop),
        "paired_mrr_gain": float(reciprocal_gain.mean()),
        "paired_mrr_gain_se": gain_se,
        "paired_mrr_gain_95_interval": [
            float(reciprocal_gain.mean() - 1.96 * gain_se),
            float(reciprocal_gain.mean() + 1.96 * gain_se)],
    }
    path = args.output if args.output.is_absolute() else ROOT / args.output
    path.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
