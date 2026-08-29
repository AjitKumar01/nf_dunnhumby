#!/usr/bin/env python3
"""Train and evaluate all verified external baselines on the locked test manifest."""
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
V4 = ROOT / "scripts" / "version4"
PY = sys.executable


def run(command, environment, dry_run=False):
    print("[baselines] " + shlex.join(map(str, command)), flush=True)
    if dry_run:
        return
    completed = subprocess.run(
        list(map(str, command)), cwd=ROOT, env=environment, check=False)
    if completed.returncode:
        raise SystemExit(completed.returncode)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=("published-1000", "smoke"),
                        default="published-1000")
    parser.add_argument("--skip-training", action="store_true",
                        help="score already-created checkpoints")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--shopper-orders", type=int, default=8192,
                        help="ordering samples used for final SHOPPER set likelihood")
    args = parser.parse_args()
    environment = os.environ.copy()
    environment["V3_AFFINITY"] = "1"
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT / "artifacts" / "native" / "lib"), str(V4),
         environment.get("PYTHONPATH", "")])
    smoke = args.profile == "smoke"
    iterations = 2 if smoke else 1000
    tag = "pipeline_smoke" if smoke else "pipeline1000"
    if not args.skip_training:
        for model in ("multinomial", "bernoulli", "dpp", "ndpp", "shopper"):
            command = [
                PY, "-u", V4 / "train_baseline_verified.py",
                "--model", model, "--tag", tag,
                "--iters", iterations, "--batch", 2 if smoke else 24,
                "--lr", 0.002, "--R", 120, "--nmax", 120,
                "--eval-every", 1 if smoke else 200,
                "--n-val", 16 if smoke else 512,
                "--eval-chunk", 2 if smoke else 8,
                "--eval-orders", 8 if smoke else 512,
            ]
            run(command, environment, args.dry_run)
    run([
        PY, "-u", V4 / "audit_other_baselines_fair.py",
        "--full-per-trip", "reports/likelihood_test_per_trip.npz",
        "--full-key", "target_child", "--split", "test",
        "--iteration", iterations, "--baseline-tag", tag,
        "--shopper-orders", 8 if smoke else args.shopper_orders,
        "--maximum-trips", 16 if smoke else 0,
        "--output", "reports/baselines",
    ], environment, args.dry_run)


if __name__ == "__main__":
    main()
