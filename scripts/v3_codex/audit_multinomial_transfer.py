"""Test whether the multinomial gap lives in its transferable additive utility blocks."""
import argparse
import copy
import json
import os

import numpy as np
import torch

import baselines as BL
import baselines2 as B2
from audit_initialization import score_main, score_multinomial
from bench_same_trips import OUT, load_main, strict_load, summarize
from data import build
from features import Features
from fit import Batcher


def transfer_nonprice(main, baseline):
    """Copy the common utility family while respecting RaggedModel's centred gauges.

    LinearIndex uses raw household/week/store factors.  RaggedModel centres the context
    factor, so the removed mean is an item-specific constant and belongs in ``lam``.
    Price is deliberately retained from the main model: its non-negative coefficient and
    common/relative split are not equivalent to the baseline's unconstrained dot product.
    """
    source = baseline.idx
    with torch.no_grad():
        main.alpha.copy_(source.alpha)
        main.theta.copy_(source.theta)
        main.mu.copy_(source.mu)
        week_mean = source.delta.mean(0)
        main.delta[:source.delta.shape[0]].copy_(source.delta)
        if main.delta.shape[0] > source.delta.shape[0]:
            main.delta[source.delta.shape[0]:].copy_(week_mean)
        main.zeta.copy_(source.zeta)
        main.xi.copy_(source.xi)
        main.w_dsp.copy_(source.w_dsp)
        main.w_mlr.copy_(source.w_mlr)
        main.lam.copy_(source.lam
                       + source.alpha @ source.theta.mean(0)
                       + source.mu @ week_mean
                       + source.zeta @ source.xi.mean(0))


@torch.no_grad()
def verify_nonprice_mapping(main, baseline, mb, bb, trips):
    md = mb.make(trips)
    bd = bb.make(trips)
    ix, ctx, _, house, *_ = md
    main.house, main.ctx = house, ctx
    zero_base = {key: value for key, value in bd["ctx"].items()}
    zero_main = {key: value for key, value in ctx.items()}
    zero_base["dlp"] = torch.zeros_like(zero_base["dlp"])
    zero_main["dlp"] = torch.zeros_like(zero_main["dlp"])
    if "dlp_bar" in zero_main:
        zero_main["dlp_bar"] = torch.zeros_like(zero_main["dlp_bar"])
    got = main.b_at(ix.item, ix.item_trip, zero_main)
    want = baseline.idx(bd["item"], bd["st"], bd["house"], zero_base)
    if not torch.equal(ix.item, bd["item"]) or not torch.equal(ix.item_trip, bd["st"]):
        raise RuntimeError("main and baseline assortment slots differ")
    return float((got - want).abs().max())


def parts(values):
    return {name: summarize(values[name], values["lines"])
            for name in ("joint", "size", "composition")}


def main(args):
    torch.set_default_dtype(torch.float64)
    D = build()
    J, N, S = (int(D[k]) for k in ("n_item", "n_user", "n_store"))
    features = Features(J, S, 712)
    mb, bb = Batcher(D, features, args.nmax), BL.Batches(D, features)
    fitted, _, _ = load_main(os.path.join(OUT, args.main_checkpoint),
                              D, 0, args.nmax, args.R)
    law = B2.size_law(D, args.nmax, args.R)
    baseline = B2.Multinomial(J, N, S, law, K=32, Kp=8).double().eval()
    strict_load(baseline, os.path.join(OUT, args.multinomial_checkpoint),
                ignore=("log_pn",))

    hybrid = copy.deepcopy(fitted)
    transfer_nonprice(hybrid, baseline)
    mapping_error = verify_nonprice_mapping(hybrid, baseline, mb, bb,
                                             np.flatnonzero(D["trip_split"] == 1)[:2])
    if mapping_error > 1e-10:
        raise RuntimeError(f"gauge transfer does not reproduce non-price utility: {mapping_error}")

    validation = np.flatnonzero(D["trip_split"] == 1)
    validation = validation[np.random.default_rng(12345).permutation(len(validation))]
    validation = validation[:args.n_val]
    result = {"n": len(validation), "mapping_max_abs": mapping_error, "models": {}}
    for name, model in (("main", fitted), ("transferred_nonprice", hybrid)):
        print(f"[transfer] scoring {name}", flush=True)
        result["models"][name] = parts(score_main(model, mb, validation, args.main_chunk))
    print("[transfer] scoring multinomial", flush=True)
    result["models"]["multinomial"] = parts(
        score_multinomial(baseline, bb, validation, args.R, args.baseline_chunk))
    path = os.path.join(OUT, args.output + ".json")
    with open(path, "w") as stream:
        json.dump(result, stream, indent=2)
    for name, score in result["models"].items():
        print(f"[transfer] {name:22s} joint {score['joint']['per_basket']:9.4f}  "
              f"size {score['size']['per_basket']:8.4f}  "
              f"composition {score['composition']['per_basket']:9.4f}", flush=True)
    print(f"[transfer] wrote {path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--main-checkpoint", default="v3_run112_blockscaled_safe_bestmrr.pt")
    parser.add_argument("--multinomial-checkpoint", default="v3_bl_multinom.pt")
    parser.add_argument("--n-val", type=int, default=192)
    parser.add_argument("--nmax", type=int, default=120)
    parser.add_argument("--R", type=int, default=23)
    parser.add_argument("--main-chunk", type=int, default=24)
    parser.add_argument("--baseline-chunk", type=int, default=8)
    parser.add_argument("--output", default="v3_multinomial_transfer_audit")
    main(parser.parse_args())
