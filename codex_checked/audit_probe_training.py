"""Explain the first-update validation regression in the fresh QMC probes.

The probe starts from an exposure-corrected popularity intercept that was estimated from
the whole training split.  Adam then sees only 24 baskets per update.  This audit scores
the exact fit.py validation manifest, splits joint set likelihood into size and conditional
composition, and swaps fitted parameter blocks back to their iteration-zero values.  The
hybrids identify which update block is responsible without relying on loss-scale guesses.
"""
import argparse
import copy
import json
import os

import numpy as np
import torch

from audit_initialization import initialize_size, score_main
from bench_same_trips import OUT, summarize
from data import build
from features import Features
from fit import Batcher, popularity_logits
from ragged import RaggedModel, set_quad


SET_UTILITY = (
    "alpha", "theta", "price_kappa", "gamma", "beta", "w_dsp", "w_mlr",
    "mu", "delta", "psi", "zeta", "xi",
)
INTERACTION = ("phi", "rho_c")
UNITS = ("a_q", "gamma_q", "beta_q", "log_r")


def in_support(D, trips, nmax, category_cap):
    keep = np.ones(len(trips), dtype=bool)
    ptr, category = D["line_ptr"], D["line_cat"]
    for k, trip in enumerate(trips):
        lo, hi = int(ptr[trip]), int(ptr[trip + 1])
        keep[k] = (hi - lo <= nmax and
                   (hi == lo or np.bincount(category[lo:hi]).max() <= category_cap))
    return trips[keep]


def make_initial(D, batcher, training, args):
    J, N, C, S = (int(D[k]) for k in ("n_item", "n_user", "n_cat", "n_store"))
    torch.manual_seed(args.seed)
    model = RaggedModel(J, N, C, K=32, Kz=32, nmax=args.nmax, R=args.R,
                        S=S, Kp=8, phi_init=0.03, taste_init=0.03).double()
    with torch.no_grad():
        category = torch.zeros(J, dtype=torch.long)
        category[torch.as_tensor(D["line_item"], dtype=torch.long)] = \
            torch.as_tensor(D["line_cat"], dtype=torch.long)
        model.cat_of.copy_(category)
        model.lam.copy_(popularity_logits(D, training))
        model.psi.zero_()
    set_quad(model, qmc_n=32, qmc_seed=0, qmc_reps=4, Kz=32, probe=-1,
             steps=2, chunk=32, size_bands=1, size_steps=2,
             mode_logtol=8.0, mode_sep=1.0)
    initialize_size(model, D, training, batcher, args.nmax)
    return model.eval()


def load_model(path, template):
    model = copy.deepcopy(template)
    blob = torch.load(path, map_location="cpu", weights_only=False)
    state = blob["model"] if isinstance(blob, dict) and "model" in blob else blob
    model.load_state_dict(state, strict=True)
    return model.eval(), blob


def replace(model, initial, names):
    source = dict(initial.named_parameters())
    target = dict(model.named_parameters())
    with torch.no_grad():
        for name in names:
            target[name].copy_(source[name])


def score(model, batcher, trips, chunk):
    values = score_main(model, batcher, trips, chunk)
    return {part: summarize(values[part], values["lines"])
            for part in ("joint", "size", "composition")}


def parameter_drift(initial, fitted):
    ans = {}
    p0, p1 = dict(initial.named_parameters()), dict(fitted.named_parameters())
    for name in p0:
        delta = p1[name].detach() - p0[name].detach()
        ans[name] = dict(rms=float(delta.square().mean().sqrt()),
                         max_abs=float(delta.abs().max()),
                         initial_sd=float(p0[name].detach().std()) if p0[name].numel() > 1 else 0.0,
                         fitted_sd=float(p1[name].detach().std()) if p1[name].numel() > 1 else 0.0)
    return ans


def main(args):
    torch.set_default_dtype(torch.float64)
    D = build()
    J, S = int(D["n_item"]), int(D["n_store"])
    batcher = Batcher(D, Features(J, S, 712), args.nmax)
    training = in_support(D, np.flatnonzero(D["trip_split"] == 0), args.nmax, args.R)
    validation = in_support(D, np.flatnonzero(D["trip_split"] == 1), args.nmax, args.R)
    validation = validation[np.random.default_rng(12345).permutation(len(validation))]
    validation = validation[:args.n_val]
    train_eval = training[np.random.default_rng(12345).choice(
        len(training), size=args.n_val, replace=False)]

    initial = make_initial(D, batcher, training, args)
    fitted, blob = load_model(os.path.join(OUT, args.checkpoint), initial)
    models = {"initial": initial, "fitted": fitted}
    for label, names in {
        "fitted_reset_lam": ("lam",),
        "fitted_reset_size": ("rho_0_free",),
        "fitted_reset_utility": SET_UTILITY,
        "fitted_reset_interaction": INTERACTION,
        "fitted_reset_units": UNITS,
        "initial_fitted_lam": ("lam",),
    }.items():
        if label == "initial_fitted_lam":
            hybrid = copy.deepcopy(initial)
            replace(hybrid, fitted, names)
        else:
            hybrid = copy.deepcopy(fitted)
            replace(hybrid, initial, names)
        models[label] = hybrid.eval()

    result = {
        "checkpoint": args.checkpoint,
        "iteration": blob.get("iter") if isinstance(blob, dict) else None,
        "n_eval": args.n_val,
        "splits": {"train": {}, "valid": {}},
        "parameter_drift": parameter_drift(initial, fitted),
    }
    for split, trips in (("train", train_eval), ("valid", validation)):
        for label, model in models.items():
            print(f"[drift] {split:5s} {label}", flush=True)
            result["splits"][split][label] = score(model, batcher, trips, args.chunk)

    path = os.path.join(OUT, args.output + ".json")
    with open(path, "w") as stream:
        json.dump(result, stream, indent=2)
    for split in ("train", "valid"):
        print(f"[drift] {split.upper()}", flush=True)
        for label, parts in result["splits"][split].items():
            print(f"[drift] {label:26s} joint {parts['joint']['per_basket']:9.4f}  "
                  f"size {parts['size']['per_basket']:8.4f}  "
                  f"composition {parts['composition']['per_basket']:9.4f}", flush=True)
    print(f"[drift] wrote {path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="v3_run115_popinit_lr5_probe.pt")
    parser.add_argument("--output", default="v3_run115_training_audit")
    parser.add_argument("--n-val", type=int, default=384)
    parser.add_argument("--nmax", type=int, default=120)
    parser.add_argument("--R", type=int, default=23)
    parser.add_argument("--chunk", type=int, default=24)
    parser.add_argument("--seed", type=int, default=0)
    main(parser.parse_args())
