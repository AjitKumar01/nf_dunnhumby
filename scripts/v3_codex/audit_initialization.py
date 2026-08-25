"""Audit iteration-zero likelihoods on one fixed manifest.

This deliberately scores *untrained* models.  The historical baseline JSON files contain
post-training values (and their old scripts first logged at iteration 2,000), so comparing
those values with the main trainer's iteration-100/200 line confounds initialization,
learning rate, and training exposure.
"""
import argparse
import copy
import json
import os

import numpy as np
import torch

import baselines as BL
import baselines2 as B2
import evalall as EA
from bench_same_trips import OUT, strict_load, summarize
from data import build
from features import Features
from fit import Batcher, _size_coeffs, popularity_logits
from ragged import RaggedModel, set_quad


def initialize_size(model, D, training, batcher, nmax):
    """Reproduce fit.py's context-reference empirical-size initialization."""
    sizes = D["trip_nlines"][training]
    count = np.bincount(np.clip(sizes, 0, nmax), minlength=nmax + 1) + 0.5
    target = torch.log(torch.as_tensor(count / count.sum(), dtype=torch.float64))
    sub = training[np.random.default_rng(0).choice(len(training), size=64, replace=False)]
    ix, ctx, _, house, *_ = batcher.make(sub)
    model.house, model.ctx = house, ctx
    z = torch.zeros(ix.B, 1, model.Kz, dtype=torch.float64)
    with torch.no_grad():
        coeff = _size_coeffs(model, z, ix)
        rho = coeff[:nmax + 1] - target
        model.rho_0_free.copy_((rho - rho[0])[1:])


@torch.no_grad()
def score_main(model, batcher, trips, chunk):
    joint, size, composition, lines = [], [], [], []
    for start in range(0, len(trips), chunk):
        sub = trips[start:start + chunk]
        ix, ctx, lctx, house, li, lt, lc, _ = batcher.make(sub)
        model.house, model.ctx = house, ctx
        ll, pn = model.loglik(ix, li, lt, lc, line_ctx=lctx, return_size=True)
        n = torch.bincount(lt, minlength=len(sub))
        lpn = torch.log(pn[torch.arange(len(sub)), n - 1].clamp_min(1e-300))
        joint.extend(ll.tolist()); size.extend(lpn.tolist())
        composition.extend((ll - lpn).tolist()); lines.extend(n.tolist())
    return dict(joint=np.asarray(joint), size=np.asarray(size),
                composition=np.asarray(composition), lines=np.asarray(lines))


@torch.no_grad()
def score_multinomial(model, batcher, trips, R, chunk):
    joint, size, composition, lines = [], [], [], []
    for start in range(0, len(trips), chunk):
        sub = trips[start:start + chunk]
        d = batcher.make(sub)
        ll = model.loglik(d, category_cap=R)
        n = torch.bincount(d["lt"], minlength=len(sub))
        lpn = model.log_pn[n]
        joint.extend(ll.tolist()); size.extend(lpn.tolist())
        composition.extend((ll - lpn).tolist()); lines.extend(n.tolist())
    return dict(joint=np.asarray(joint), size=np.asarray(size),
                composition=np.asarray(composition), lines=np.asarray(lines))


def make_main(D, batcher, training, a):
    J, N, C, S = (int(D[k]) for k in ("n_item", "n_user", "n_cat", "n_store"))
    model = RaggedModel(J, N, C, K=32, Kz=32, nmax=a.nmax, R=a.R,
                        S=S, Kp=8, phi_init=0.03).double()
    with torch.no_grad():
        cat = torch.zeros(J, dtype=torch.long)
        cat[torch.as_tensor(D["line_item"], dtype=torch.long)] = \
            torch.as_tensor(D["line_cat"], dtype=torch.long)
        model.cat_of.copy_(cat)
    set_quad(model, qmc_n=a.qmc_n, qmc_seed=0, qmc_reps=4, Kz=32,
             probe=-1, steps=2, chunk=a.qmc_n, size_bands=0)
    return model


def main(a):
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(0)
    D = build()
    J, N, S = (int(D[k]) for k in ("n_item", "n_user", "n_store"))
    features = Features(J, S, 712)
    main_batcher, base_batcher = Batcher(D, features, a.nmax), BL.Batches(D, features)
    training = np.flatnonzero(D["trip_split"] == 0)
    trips = EA.sample_split(D, "valid", a.n_trips, a.nmax, a.R, seed=a.seed)
    pop = popularity_logits(D, training)

    template = make_main(D, main_batcher, training, a)
    raw = copy.deepcopy(template)
    initialize_size(raw, D, training, main_batcher, a.nmax)

    low_noise = copy.deepcopy(template)
    with torch.no_grad():
        low_noise.alpha.mul_(0.1); low_noise.theta.mul_(0.1)
    initialize_size(low_noise, D, training, main_batcher, a.nmax)

    pop_model = copy.deepcopy(template)
    with torch.no_grad():
        pop_model.lam.copy_(pop)
        pop_model.alpha.mul_(0.1); pop_model.theta.mul_(0.1)
    initialize_size(pop_model, D, training, main_batcher, a.nmax)

    pop_additive = copy.deepcopy(pop_model)
    with torch.no_grad():
        pop_additive.phi.zero_(); pop_additive.rho_c.zero_()
    pop_additive._exact_additive = True
    initialize_size(pop_additive, D, training, main_batcher, a.nmax)

    # Diagnostic frequency-only limit: no random bilinear utility at all.  It is not a
    # trainable initialization because zeroing both sides kills their gradients.
    pop_only = copy.deepcopy(pop_additive)
    with torch.no_grad():
        for name in ("alpha", "theta", "mu", "delta", "zeta", "xi", "w_dsp", "w_mlr"):
            getattr(pop_only, name).zero_()
    initialize_size(pop_only, D, training, main_batcher, a.nmax)

    variants = dict(current_random=raw, low_taste_noise=low_noise,
                    popularity_low_noise=pop_model,
                    popularity_additive=pop_additive, popularity_only=pop_only)
    result = dict(seed=a.seed, n_trips=len(trips), popularity_lam_sd=float(pop.std()),
                  variants={})
    for name, model in variants.items():
        print(f"[init] scoring {name}", flush=True)
        values = score_main(model.eval(), main_batcher, trips, a.main_chunk)
        result["variants"][name] = {
            part: summarize(values[part], values["lines"])
            for part in ("joint", "size", "composition")}

    law = B2.size_law(D, a.nmax, a.R)
    for name, checkpoint in (("multinomial_random", ""),
                             ("multinomial_trained_legacy", a.multinomial_ckpt)):
        model = B2.Multinomial(J, N, S, law, K=32, Kp=8).double()
        if checkpoint:
            strict_load(model, os.path.join(OUT, checkpoint), ignore=("log_pn",))
        print(f"[init] scoring {name}", flush=True)
        values = score_multinomial(model.eval(), base_batcher, trips, a.R, a.base_chunk)
        result["variants"][name] = {
            part: summarize(values[part], values["lines"])
            for part in ("joint", "size", "composition")}

    path = os.path.join(OUT, a.output + ".json")
    with open(path, "w") as stream:
        json.dump(result, stream, indent=2)
    for name, parts in result["variants"].items():
        print(f"[init] {name:27s} joint {parts['joint']['per_basket']:9.3f}  "
              f"size {parts['size']['per_basket']:7.3f}  "
              f"composition {parts['composition']['per_basket']:9.3f}", flush=True)
    print(f"[init] wrote {path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trips", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--nmax", type=int, default=120)
    parser.add_argument("--R", type=int, default=23)
    parser.add_argument("--qmc-n", type=int, default=32)
    parser.add_argument("--main-chunk", type=int, default=24)
    parser.add_argument("--base-chunk", type=int, default=8)
    parser.add_argument("--multinomial-ckpt", default="v3_bl_multinom.pt")
    parser.add_argument("--output", default="v3_initialization_audit")
    main(parser.parse_args())
