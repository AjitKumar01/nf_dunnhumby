#!/usr/bin/env python3
"""Create the untrained, reproducible Version-4 initialization artifact.

This stage performs no optimizer update and loads no checkpoint.  The additive fit that
follows is exact, so constructing an adaptive quadrature rule here would add cost without
changing its objective.  The artifact uses the existing checksummed container format; its
single placeholder index is compatibility metadata and is never used by the staged
pipeline.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("V3_AFFINITY", "1")

import numpy as np
import torch

from data import build
from features import Features
from fit import (Batcher, calibrate_size_ipf, initialize_size_potential,
                 initialize_taste_moments, popularity_logits)
from ragged import RaggedModel
from sparse_artifact import (initialize_nested_trace_class_phi,
                             save_sparse_initialization_artifact)


torch.set_default_dtype(torch.float64)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path,
                        default=Path("artifacts/initialization.pt"))
    parser.add_argument("--manifest", type=Path,
                        default=Path("artifacts/initialization.json"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--K", type=int, default=32)
    parser.add_argument("--Kz", type=int, default=32)
    parser.add_argument("--Kp", type=int, default=8)
    parser.add_argument("--active-rank", type=int, default=8)
    parser.add_argument("--nmax", type=int, default=120)
    parser.add_argument("--R", type=int, default=120)
    parser.add_argument("--ipf-trips", type=int, default=96)
    parser.add_argument("--ipf-steps", type=int, default=0,
                        help="optional initialization-only size IPF; the default leaves "
                             "all size learning to exact additive maximum likelihood")
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    data = build()
    dimensions = tuple(int(data[key]) for key in
                       ("n_item", "n_user", "n_cat", "n_store"))
    products, households, categories, stores = dimensions
    if products == 5455 and categories != 280:
        raise RuntimeError("Version-4 requires the 280-row affinity partition")
    training = np.flatnonzero(data["trip_split"] == 0)
    model = RaggedModel(products, households, categories, K=args.K, Kz=args.Kz,
                        nmax=args.nmax, R=args.R, seed=args.seed, S=stores,
                        Kp=args.Kp, phi_init=0.0, taste_init=0.03)
    category = torch.zeros(products, dtype=torch.long)
    category[torch.as_tensor(data["line_item"], dtype=torch.long)] = \
        torch.as_tensor(data["line_cat"], dtype=torch.long)
    with torch.no_grad():
        model.cat_of.copy_(category)
        model.lam.copy_(popularity_logits(data, training))
        model.psi.zero_()
    initialize_taste_moments(model, data, training, strength=1.0,
                             prior=100.0, clip=3.0, seed=args.seed)
    # Stored only to bind the declared maximum starting rank.  The exact additive stage
    # explicitly zeros Phi before its first objective evaluation.
    initialize_nested_trace_class_phi(model, active_rank=args.active_rank,
                                      row_rms=0.03, decay=0.84, seed=823)
    batcher = Batcher(data, Features(products, stores, 712), args.nmax)
    initialize_size_potential(model, data, training, batcher, args.nmax)
    if args.ipf_steps:
        calibrate_size_ipf(model, data, training, batcher, args.nmax,
                           steps=args.ipf_steps, n_trips=args.ipf_trips,
                           chunk=24, damp=0.7)

    metadata = {
        "J": products, "N": households, "C": categories, "S": stores,
        "K": args.K, "Kz": args.Kz, "Kp": args.Kp,
        "nmax": args.nmax, "R": args.R, "seed": args.seed,
        "active_rank": args.active_rank, "affinity_partition": True,
        "initialization_only": True, "no_rec": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary = save_sparse_initialization_artifact(
        args.output, model, metadata=metadata,
        sequence=[tuple([1] * args.Kz)], calibration_trips=[])
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(summary, indent=2, default=str) + "\n")
    print(f"[initialize] wrote fresh artifact {args.output}")


if __name__ == "__main__":
    main()
