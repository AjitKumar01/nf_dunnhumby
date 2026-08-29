#!/usr/bin/env python3
"""One reproducible driver for the selected Version-4 staged pipeline."""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
V4 = ROOT / "scripts" / "version4"
ART = ROOT / "artifacts"
REPORT = ROOT / "reports"


def command_text(command: list[str]) -> str:
    return shlex.join(command)


class Driver:
    def __init__(self, dry_run: bool):
        self.dry_run = dry_run
        self.commands: list[list[str]] = []
        self.environment = os.environ.copy()
        native = ART / "native" / "lib"
        old = self.environment.get("PYTHONPATH", "")
        self.environment["PYTHONPATH"] = os.pathsep.join(
            [str(native), str(V4)] + ([old] if old else []))
        self.environment["V3_AFFINITY"] = "1"

    def run(self, command: list[str], *, allow_failure: bool = False) -> int:
        self.commands.append(command)
        print(f"[pipeline] {command_text(command)}", flush=True)
        if self.dry_run:
            return 0
        result = subprocess.run(command, cwd=ROOT, env=self.environment,
                                check=False)
        if result.returncode and not allow_failure:
            raise SystemExit(result.returncode)
        return result.returncode


def script(name: str, *arguments: object) -> list[str]:
    return [PY, "-u", str(V4 / name), *map(str, arguments)]


def rank_selection(driver: Driver, parent: Path, contexts: int) -> tuple[int, Path]:
    if driver.dry_run:
        output = ART / "interaction_basis_rank7.npz"
        driver.run(script("build_spectral_phi_initialization.py", "--parent", parent,
                          "--trips", contexts, "--draws", 2, "--rank", 8,
                          "--output", ART / "interaction_basis_rank8.npz"))
        driver.run(script("build_spectral_phi_initialization.py", "--parent", parent,
                          "--trips", contexts, "--draws", 2, "--rank", 7,
                          "--output", output))
        return 7, output
    for rank in range(8, 3, -1):
        output = ART / f"interaction_basis_rank{rank}.npz"
        driver.run(script("build_spectral_phi_initialization.py", "--parent", parent,
                          "--trips", contexts, "--draws", 2, "--rank", rank,
                          "--output", output), allow_failure=True)
        report = output.with_suffix(".json")
        if report.exists() and json.loads(report.read_text()).get(
                "stable_for_scale_profile"):
            print(f"[pipeline] selected independently stable rank {rank}")
            return rank, output
    raise SystemExit("no rank in 4..8 passed the predeclared split-half gate")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--from-raw", action="store_true",
                        help="rebuild data/ and basket_input/ from dunnhumby CSVs")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the complete command graph without executing it")
    parser.add_argument("--profile", choices=("full", "smoke"), default="full")
    parser.add_argument("--stop-after", choices=(
        "data", "initialize", "additive", "rank", "interaction", "size",
        "evaluation", "certification"), default="certification")
    args = parser.parse_args()
    ART.mkdir(exist_ok=True)
    REPORT.mkdir(exist_ok=True)
    (ROOT / "out").mkdir(exist_ok=True)
    driver = Driver(args.dry_run)

    if args.from_raw:
        driver.run([PY, "-u", "scripts/data/01_build_base.py"])
        driver.run([PY, "-u", "scripts/data/22_basket_data.py"])
        driver.run([PY, "-u", "scripts/data/23_promo_data.py"])
    driver.run(script("data.py", "--force"))
    driver.run(script("build_affinity_partition.py"))
    # Rebuild the ragged index after switching from commodity to affinity categories.
    driver.run(script("data.py", "--force"))
    if args.stop_after == "data":
        return

    driver.run([PY, str(V4 / "setup_poly_degree_native.py"), "build_ext",
                "--build-lib", str(ART / "native" / "lib"),
                "--build-temp", str(ART / "native" / "temp")])
    initialization = ART / "initialization.pt"
    driver.run(script("initialize_version4.py", "--output", initialization,
                      "--manifest", ART / "initialization.json"))
    if args.stop_after == "initialize":
        return

    full = args.profile == "full"
    additive_iterations = 12000 if full else 10
    driver.run(script(
        "fit_exact_additive.py", "--artifact", initialization,
        "--label", "pipeline_additive", "--iters", additive_iterations,
        "--batch", 128 if full else 8, "--lr", 0.002,
        "--weight-decay", 1e-5, "--validation-trips", 1024 if full else 16,
        "--validation-chunk", 128 if full else 8,
        "--eval-every", 100 if full else 5, "--size-kl", 1.0,
        "--lr-patience", 4, "--lr-factor", 0.5, "--min-lr", 6.25e-5,
        "--convergence-patience", 8,
        "--convergence-min-updates", 4000 if full else 10,
        "--validation-min-delta", 0.001,
        "--rkl-w", 10.0, "--elast-w", 20.0, "--elast-target", -0.121,
        "--pool-prod", 1.45, "--lam-centre", 1, "--seed", 29001))
    additive = ROOT / "out" / "v3_pipeline_additive_best.pt"
    if args.stop_after == "additive":
        return

    rank, basis = rank_selection(driver, additive, 50000 if full else 128)
    if args.stop_after == "rank":
        return

    interaction = ART / "interaction.pt"
    driver.run(script(
        "fit_projected_fisher_interactions.py", "--parent", additive,
        "--spectral", basis, "--contexts", 0 if full else 128,
        "--draws", 2, "--batch", 128 if full else 16, "--rank", rank,
        "--score-mass", 1.0, "--spectral-max", 1.0,
        "--minimum-crossfit-gain", 0.005 if full else -1.0,
        "--output", interaction))
    if args.stop_after == "interaction":
        return

    candidate = ART / "candidate.pt"
    driver.run(script(
        "calibrate_projected_fisher_size.py", "--parent", additive,
        "--child", interaction, "--contexts", 10000 if full else 64,
        "--draws", 32 if full else 4, "--batch", 96 if full else 8,
        "--minimum-crossfit-gain", 0.002 if full else -1.0,
        "--output", candidate))
    if args.stop_after == "size":
        return

    # All claims use fixed panels and complete support; recommendation is read-only.
    driver.run(script(
        "compare_rank8_parent_likelihood.py", "--parent", additive,
        "--child", candidate, "--split", "validation", "--trips", 4096 if full else 16,
        "--rank", rank, "--target-level", rank + 2,
        "--audit-trips", 64 if full else 4,
        "--output", REPORT / "likelihood_validation.json"))
    driver.run(script(
        "compare_rank8_parent_likelihood.py", "--parent", additive,
        "--child", candidate, "--split", "test", "--trips", 4096 if full else 16,
        "--rank", rank, "--target-level", rank + 2,
        "--audit-trips", 64 if full else 4,
        "--output", REPORT / "likelihood_test.json"))
    driver.run(script(
        "eval_smolyak_rank8_mrr.py", "--ckpt", candidate, "--split", "test",
        "--trips", 2000 if full else 16, "--rank", rank,
        "--level", rank + 2, "--output", REPORT / "recommendation.json"))
    driver.run(script(
        "audit_particle_counterfactual_generation.py", "--ckpt", candidate,
        "--trips", 64 if full else 2, "--particles", 64 if full else 4,
        "--output", REPORT / "generation_counterfactual.json"))
    driver.run(script(
        "audit_customer_segments.py", "--ckpt", candidate,
        "--candidate-segments", 3, 4, 5, 6,
        "--contexts-per-segment", 48 if full else 2,
        "--particles", 32 if full else 4,
        "--output", REPORT / "customer_segments.json",
        "--assignments", ART / "customer_segments.npz"))
    if args.stop_after == "evaluation":
        return

    driver.run(script(
        "audit_population_size.py", "--checkpoint", candidate,
        "--rank", rank, "--screen-level", rank + 1,
        "--confirm-level", rank + 2,
        "--confirm-contexts", 96 if full else 8,
        "--output", REPORT / "population_size.json"))
    if args.dry_run:
        print("[pipeline] dry run complete; no stage was executed")
    else:
        print("[pipeline] certification passed; candidate.pt is the accepted model")


if __name__ == "__main__":
    main()
