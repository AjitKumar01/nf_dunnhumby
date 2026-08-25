"""Paired likelihood ablation of phi and rho_c on the shared evaluation trips."""
import argparse
import json
import os

import numpy as np
import torch

import evalall as EA
from bench_same_trips import OUT, load_main, paired, summarize
from data import build
from decompose_same_trips import score_main
from features import Features
from fit import Batcher


def main(a):
    torch.set_default_dtype(torch.float64)
    D = build()
    J, _, _, S = (int(D[k]) for k in ("n_item", "n_user", "n_cat", "n_store"))
    batcher = Batcher(D, Features(J, S, 712), a.nmax)
    model, _, meta = load_main(os.path.join(OUT, a.main_ckpt), D, 0, a.nmax, a.R)
    phi0, rhoc0 = model.phi.detach().clone(), model.rho_c.detach().clone()
    variants = {
        "full": (False, False),
        "phi_zero": (True, False),
        "rho_c_zero": (False, True),
        "all_interactions_zero": (True, True),
    }
    result = {"checkpoint": a.main_ckpt, "meta": meta, "seed": a.seed, "splits": {}}
    for split in a.splits.split(","):
        trips = EA.sample_split(D, split, a.n_trips, a.nmax, a.R, seed=a.seed)
        scores = {}
        for name, (zero_phi, zero_rhoc) in variants.items():
            with torch.no_grad():
                model.phi.copy_(torch.zeros_like(phi0) if zero_phi else phi0)
                model.rho_c.copy_(torch.zeros_like(rhoc0) if zero_rhoc else rhoc0)
            print(f"[abl] {split}: {name}", flush=True)
            scores[name] = score_main(model, batcher, trips, a.chunk)
        block = {}
        for name, values in scores.items():
            block[name] = {}
            for part in ("joint", "size", "composition"):
                block[name][part] = summarize(values[part], values["lines"])
                if name != "full":
                    block[name][part]["full_minus_ablation"] = paired(
                        scores["full"][part], values[part])
            print(f"[abl] {split:5s} {name:22s} joint {values['joint'].mean():9.4f}  "
                  f"composition {values['composition'].mean():9.4f}", flush=True)
        result["splits"][split] = block
    with torch.no_grad():
        model.phi.copy_(phi0); model.rho_c.copy_(rhoc0)
    path = os.path.join(OUT, a.output + ".json")
    with open(path, "w") as stream:
        json.dump(result, stream, indent=2)
    print(f"[abl] wrote {path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--main-ckpt", default="v3_run112_blockscaled_safe_best.pt")
    parser.add_argument("--splits", default="valid,test")
    parser.add_argument("--n-trips", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--nmax", type=int, default=120)
    parser.add_argument("--R", type=int, default=23)
    parser.add_argument("--chunk", type=int, default=24)
    parser.add_argument("--output", default="v3_interaction_ablation")
    main(parser.parse_args())
