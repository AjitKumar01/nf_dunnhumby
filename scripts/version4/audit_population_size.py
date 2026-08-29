#!/usr/bin/env python3
"""Fail-closed full-population basket-size and Smolyak-tail audit."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from audit_particle_counterfactual_generation import ROOT, load_checkpoint
from data import build
from features import Features
from fit import Batcher
from fit_interaction_particles import supported_trips
from fit_multifidelity_rank8 import rule
from profile_rho0_size_likelihood import collect_size_law


torch.set_default_dtype(torch.float64)


def metrics(log_probability: np.ndarray, observed: np.ndarray) -> dict:
    probability = np.exp(log_probability)
    size = np.arange(1, probability.shape[1] + 1, dtype=np.float64)
    mean = probability @ size
    tail = probability[:, 59:].sum(1)
    observed_tail = np.asarray(observed) >= 60
    low_observed = np.asarray(observed) < 40
    return {
        "contexts": int(len(observed)),
        "observed_mean": float(np.mean(observed)),
        "model_mean": float(np.mean(mean)),
        "observed_tail_rate_ge_60": float(np.mean(observed_tail)),
        "model_tail_rate_ge_60": float(np.mean(tail)),
        "contexts_expected_size_ge_40": int(np.sum(mean >= 40)),
        "contexts_expected_size_ge_60": int(np.sum(mean >= 60)),
        "contexts_tail_probability_ge_half": int(np.sum(tail >= 0.5)),
        "maximum_conditional_mean": float(np.max(mean)),
        "maximum_tail_probability_ge_60": float(np.max(tail)),
        "maximum_tail_probability_when_observed_lt_40": float(
            np.max(tail[low_observed]) if np.any(low_observed) else 0.0),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation", "test"),
                        default="train")
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--screen-level", type=int, default=8)
    parser.add_argument("--confirm-level", type=int, default=9)
    parser.add_argument("--confirm-contexts", type=int, default=96)
    parser.add_argument("--contexts", type=int, default=0,
                        help="screen contexts; 0 means the complete supported population")
    parser.add_argument("--chunk", type=int, default=24)
    parser.add_argument("--maximum-low-observed-tail", type=float, default=0.5)
    parser.add_argument("--maximum-tail-rate-ratio", type=float, default=2.0)
    parser.add_argument("--tail-rate-slack", type=float, default=5e-4)
    parser.add_argument("--maximum-q9-q8-mean-gap", type=float, default=1.0)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--output", type=Path,
                        default=Path("reports/population_size.json"))
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    data = build()
    split_code = {"train": 0, "validation": 1, "test": 2}[args.split]
    checkpoint = args.checkpoint if args.checkpoint.is_absolute() \
        else ROOT / args.checkpoint
    model, _blob, meta = load_checkpoint(checkpoint, data)
    population = supported_trips(data, split_code, int(meta["nmax"]))
    full_population_size = len(population)
    if args.contexts < 0:
        raise ValueError("contexts must be nonnegative")
    if args.contexts:
        population = population[:min(args.contexts, len(population))]
    batcher = Batcher(data, Features(int(data["n_item"]),
                                     int(data["n_store"]), 712),
                      int(meta["nmax"]))
    observed, q8 = collect_size_law(
        model, batcher, population, rule(model, args.rank, args.screen_level),
        args.chunk, f"{args.split}-q{args.screen_level}")
    screen = metrics(q8, observed)
    probability = np.exp(q8)
    size = np.arange(1, probability.shape[1] + 1, dtype=np.float64)
    risk = probability[:, 59:].sum(1) + (probability @ size) / 120.0
    count = min(args.confirm_contexts, len(population))
    chosen_index = np.argsort(risk, kind="stable")[-count:]
    confirm_trips = population[chosen_index]
    confirm_observed, q9 = collect_size_law(
        model, batcher, confirm_trips,
        rule(model, args.rank, args.confirm_level), args.chunk,
        f"tail-q{args.confirm_level}")
    confirm = metrics(q9, confirm_observed)
    q8_mean = np.exp(q8[chosen_index]) @ size
    q9_mean = np.exp(q9) @ size
    fidelity = {
        "mean_absolute_expected_size_gap": float(np.mean(np.abs(q9_mean - q8_mean))),
        "maximum_absolute_expected_size_gap": float(np.max(np.abs(q9_mean - q8_mean))),
    }
    allowed_rate = (args.maximum_tail_rate_ratio
                    * screen["observed_tail_rate_ge_60"]
                    + args.tail_rate_slack)
    gates = {
        "population_tail_rate_calibrated":
            screen["model_tail_rate_ge_60"] <= allowed_rate,
        "no_low_observed_context_has_majority_extreme_tail":
            confirm["maximum_tail_probability_when_observed_lt_40"]
            <= args.maximum_low_observed_tail,
        "screen_rule_resolves_high_risk_expected_size":
            fidelity["maximum_absolute_expected_size_gap"]
            <= args.maximum_q9_q8_mean_gap,
    }
    result = {
        "checkpoint": str(checkpoint), "split": args.split,
        "full_population_contexts": int(full_population_size),
        "screened_complete_population": bool(len(population) == full_population_size),
        "support": "1..120", "rank": args.rank,
        "screen_level": args.screen_level,
        "confirm_level": args.confirm_level,
        "screen": screen, "high_risk_confirmation": confirm,
        "quadrature_fidelity": fidelity, "allowed_model_tail_rate": allowed_rate,
        "gates": gates, "passed": bool(all(gates.values())),
        "interpretation": (
            "q8 screens the requested population panel; full certification requests "
            "every supported context. q9 re-evaluates the highest-risk "
            "contexts. Failure blocks production certification but preserves artifacts."),
    }
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
