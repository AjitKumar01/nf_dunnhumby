#!/usr/bin/env python3
"""Fail-closed full-population basket-size and Smolyak-tail audit."""
from __future__ import annotations

import argparse
import hashlib
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
from profile_rho0_size_likelihood import collect_size_law, install


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


@torch.no_grad()
def one_size_panel(model, batcher, trips, quadrature):
    """Evaluate one panel, raising when the requested signed rule is invalid."""
    install(model, quadrature)
    ix, ctx, _line_ctx, house, _li, lt, _lc, _lq = batcher.make(trips)
    model.house, model.ctx = house, ctx
    _logz, probability = model.log_Z(ix, drop_empty=True, return_size=True)
    value = np.log(np.clip(probability.cpu().numpy(), 1e-300, None))
    value -= np.logaddexp.reduce(value, axis=1)[:, None]
    observed = torch.bincount(lt, minlength=ix.B).cpu().numpy()
    return observed, value


def resilient_size_panel(model, batcher, trips, rules, levels):
    """Use the cheap rule when valid; bisect failures and escalate only those trips."""
    try:
        observed, value = one_size_panel(model, batcher, trips, rules[0])
        return observed, value, np.full(len(trips), levels[0], dtype=np.int16)
    except FloatingPointError:
        if len(trips) > 1:
            middle = len(trips) // 2
            left = resilient_size_panel(
                model, batcher, trips[:middle], rules, levels)
            right = resilient_size_panel(
                model, batcher, trips[middle:], rules, levels)
            return tuple(np.concatenate((left[i], right[i])) for i in range(3))
        last_error = None
        for quadrature, level in zip(rules[1:], levels[1:]):
            try:
                observed, value = one_size_panel(
                    model, batcher, trips, quadrature)
                return observed, value, np.full(1, level, dtype=np.int16)
            except FloatingPointError as error:
                last_error = error
        raise FloatingPointError(
            f"all screen escalation levels failed for trip {int(trips[0])}: "
            f"{last_error}") from last_error


def screen_signature(checkpoint, population, rank, levels):
    digest = hashlib.sha256()
    with checkpoint.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    digest.update(np.ascontiguousarray(population, dtype=np.int64).tobytes())
    digest.update(f"rank={rank};levels={levels}".encode())
    return digest.hexdigest()


def resumable_screen(model, batcher, population, checkpoint, rank, levels,
                     chunk, output, label):
    """Checkpoint the population panel and resume after interruption or cancellation."""
    signature = screen_signature(checkpoint, population, rank, levels)
    prefix = output.with_name(f"{output.stem}.screen-{signature[:12]}")
    probability_path = Path(str(prefix) + ".log_probability.npy")
    observed_path = Path(str(prefix) + ".observed.npy")
    level_path = Path(str(prefix) + ".level.npy")
    progress_path = Path(str(prefix) + ".progress.json")
    shape = (len(population), int(model.nmax))
    expected = {
        "signature": signature, "contexts": len(population),
        "rank": rank, "levels": levels,
    }
    start = 0
    if progress_path.exists():
        progress = json.loads(progress_path.read_text())
        cache_exists = all(path.exists() for path in (
            probability_path, observed_path, level_path))
        if (cache_exists
                and all(progress.get(key) == value
                        for key, value in expected.items())):
            start = int(progress.get("completed_contexts", 0))
    mode = "r+" if start > 0 else "w+"
    probability = np.lib.format.open_memmap(
        probability_path, mode=mode, dtype=np.float64, shape=shape)
    observed = np.lib.format.open_memmap(
        observed_path, mode=mode, dtype=np.int16, shape=(len(population),))
    used_level = np.lib.format.open_memmap(
        level_path, mode=mode, dtype=np.int16, shape=(len(population),))
    rules = [rule(model, rank, level) for level in levels]
    for position in range(start, len(population), chunk):
        stop = min(position + chunk, len(population))
        got_observed, got_probability, got_level = resilient_size_panel(
            model, batcher, population[position:stop], rules, levels)
        observed[position:stop] = got_observed
        probability[position:stop] = got_probability
        used_level[position:stop] = got_level
        if ((position // chunk + 1) % 20 == 0 or stop == len(population)):
            probability.flush(); observed.flush(); used_level.flush()
            progress = {**expected, "completed_contexts": stop,
                        "complete": stop == len(population)}
            temporary = Path(str(progress_path) + ".tmp")
            temporary.write_text(json.dumps(progress, indent=2) + "\n")
            temporary.replace(progress_path)
            print(f"[population-screen] {label} {stop}/{len(population)}",
                  flush=True)
    levels_used, counts = np.unique(np.asarray(used_level), return_counts=True)
    escalated = np.flatnonzero(np.asarray(used_level) > levels[0])
    provenance = {
        "signature": signature,
        "cache_prefix": str(prefix),
        "resumed_from_context": start,
        "level_counts": {str(int(level)): int(count)
                         for level, count in zip(levels_used, counts)},
        "escalated_contexts": int(len(escalated)),
        "escalated_trip_ids_first_100": population[escalated[:100]].tolist(),
        "escalated_trip_ids_truncated": bool(len(escalated) > 100),
    }
    return (np.asarray(observed), np.asarray(probability),
            np.asarray(used_level), provenance)


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
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    screen_levels = [args.screen_level, args.confirm_level,
                     args.confirm_level + 1]
    observed, q8, used_level, screen_provenance = resumable_screen(
        model, batcher, population, checkpoint, args.rank, screen_levels,
        args.chunk, output, f"{args.split}-q{args.screen_level}")
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
        "screen_estimator": {
            "policy": ("use the requested screen level when every signed size mass is "
                       "positive; bisect invalid batches and escalate only invalid "
                       "contexts through confirm_level and confirm_level+1"),
            **screen_provenance,
        },
        "screen": screen, "high_risk_confirmation": confirm,
        "quadrature_fidelity": fidelity, "allowed_model_tail_rate": allowed_rate,
        "gates": gates, "passed": bool(all(gates.values())),
        "interpretation": (
            "The low rule screens the requested population panel. A context whose "
            "signed size masses are invalid is evaluated by the next positive rule "
            "rather than treated as a model probability. The confirm rule re-evaluates "
            "the highest-risk contexts. Failure blocks production certification but "
            "preserves artifacts and resumable screen state."),
    }
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
