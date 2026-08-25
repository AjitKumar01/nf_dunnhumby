"""Audited same-trip likelihood comparison on the complete store assortment.

Every model sees the same trip ids and every product carried by that trip's store. The
trip manifest, checkpoint hashes, per-basket standard errors, and paired gaps are written
to JSON so a table cannot outlive the exact data and weights that produced it.

The multinomial baseline is normalized on the main model's exact support (non-empty,
n<=nmax, and at most R products per affinity category). NDPP and SHOPPER are normalized
on the larger set of every non-empty assortment subset; that conservative support mismatch
is recorded rather than hidden.
"""
import argparse
import hashlib
import json
import os
import time

import numpy as np
import torch

import baselines as BL
import baselines2 as B2
import evalall as EA
from data import build
from features import Features
from fit import Batcher
from ragged import RaggedModel


HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.abspath(os.path.join(HERE, "..", "..", "out"))


def log(message):
    print(f"[same] {message}", flush=True)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def checkpoint_record(path, blob):
    stat = os.stat(path)
    return dict(path=os.path.basename(path), sha256=sha256(path), bytes=stat.st_size,
                mtime_ns=stat.st_mtime_ns,
                format=blob.get("format") if isinstance(blob, dict) else "legacy-state-dict",
                iteration=blob.get("iter") if isinstance(blob, dict) else None,
                training_provenance=("embedded" if isinstance(blob, dict) and blob.get("format")
                                     else "NOT EMBEDDED; strict shape/hash verification only"))


def state_dict(blob):
    return blob["model"] if isinstance(blob, dict) and "model" in blob else blob


def strict_load(model, path, ignore=()):
    blob = torch.load(path, map_location="cpu", weights_only=False)
    sd = dict(state_dict(blob))
    for key in ignore:
        sd.pop(key, None)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    missing = [key for key in missing if key not in ignore]
    if missing or unexpected:
        raise RuntimeError(f"{os.path.basename(path)} incompatible: missing={missing}, "
                           f"unexpected={unexpected}")
    return blob


def summarize(values, lines):
    values = np.asarray(values, dtype=np.float64)
    return dict(n=len(values), per_basket=float(values.mean()),
                se_per_basket=float(values.std(ddof=1) / np.sqrt(len(values))),
                per_line=float(values.sum() / np.asarray(lines).sum()))


def paired(main, baseline):
    d = np.asarray(main) - np.asarray(baseline)
    return dict(main_minus_baseline=float(d.mean()),
                paired_se=float(d.std(ddof=1) / np.sqrt(len(d))))


@torch.no_grad()
def main_scores(model, batcher, trips, chunk):
    values, lines = [], []
    for k in range(0, len(trips), chunk):
        sub = trips[k:k + chunk]
        ix, ctx, lctx, hh, li, lt, lc, _ = batcher.make(sub)
        model.house, model.ctx = hh, ctx
        ll = model.loglik(ix, li, lt, lc, line_ctx=lctx)
        values.extend(ll.detach().cpu().tolist())
        lines.extend(torch.bincount(lt, minlength=len(sub)).tolist())
    return np.asarray(values), np.asarray(lines)


@torch.no_grad()
def baseline_scores(model, batcher, trips, kind, chunk, seed=0,
                    shopper_orders=512, exact_max_n=6, category_cap=23):
    values, lines = [], []
    for k in range(0, len(trips), chunk):
        sub = trips[k:k + chunk]
        d = batcher.make(sub)
        if kind == "shopper":
            gen = torch.Generator().manual_seed(seed + 104729 * k)
            ll = model.loglik(d, n_orders=shopper_orders, gen=gen,
                              exact_max_n=exact_max_n)
        elif kind == "multinomial":
            ll = model.loglik(d, category_cap=category_cap)
        else:
            ll = model.loglik(d)
        if len(ll) != len(sub) or not torch.isfinite(ll).all():
            raise RuntimeError(f"{kind} returned non-finite or mis-sized scores")
        values.extend(ll.detach().cpu().tolist())
        lines.extend(torch.bincount(d["lt"], minlength=len(sub)).tolist())
    return np.asarray(values), np.asarray(lines)


def load_main(path, D, kz, nmax, R):
    J, N, C, S = (int(D[k]) for k in ("n_item", "n_user", "n_cat", "n_store"))
    blob = torch.load(path, map_location="cpu", weights_only=False)
    q = blob.get("quad") or {}
    actual_kz = int(q.get("Kz", state_dict(blob)["phi"].shape[1]))
    if kz and kz != actual_kz:
        raise ValueError(f"requested Kz={kz}, checkpoint has Kz={actual_kz}")
    model = RaggedModel(J=J, N=N, C=C, K=32, Kz=actual_kz,
                        nmax=nmax, R=R, S=S, Kp=8)
    meta = EA.load_any(path, model, J, D)
    model.double().eval()
    return model, blob, meta


def main(a):
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(a.seed)
    D = build()
    J, N, _, S = (int(D[k]) for k in ("n_item", "n_user", "n_cat", "n_store"))
    features = Features(J, S, 712)
    main_batcher = Batcher(D, features, a.nmax)
    baseline_batcher = BL.Batches(D, features)

    splits = [s.strip() for s in a.splits.split(",") if s.strip()]
    picks = {s: EA.sample_split(D, s, a.n_trips, a.nmax, a.R, seed=a.seed)
             for s in splits}
    manifest = {}
    for split, trips in picks.items():
        raw = np.ascontiguousarray(trips, dtype=np.int64).tobytes()
        weeks = D["trip_week"][trips]
        manifest[split] = dict(n=len(trips), sha256=hashlib.sha256(raw).hexdigest(),
                               first_ids=[int(x) for x in trips[:10]],
                               weeks=[int(weeks.min()), int(weeks.max())],
                               distinct_weeks=int(len(np.unique(weeks))))
        log(f"{split}: {len(trips)} identical in-support trips, id hash "
            f"{manifest[split]['sha256'][:12]}, weeks {weeks.min()}-{weeks.max()}")

    main_path = os.path.join(OUT, a.main_ckpt)
    model, main_blob, main_meta = load_main(main_path, D, a.Kz, a.nmax, a.R)
    records = {"main": checkpoint_record(main_path, main_blob)}
    records["main"]["meta"] = main_meta

    law = B2.size_law(D, a.nmax, a.R)
    specs = {
        "multinomial": (B2.Multinomial(J, N, S, law, K=32, Kp=8),
                         os.path.join(OUT, a.multinomial_ckpt), ("log_pn",)),
        "ndpp": (B2.NDPP(J, N, S, rank=16, srank=8, K=32, Kp=8),
                 os.path.join(OUT, a.ndpp_ckpt), ()),
        "shopper": (B2.Shopper(J, N, S, K=32, Kp=8),
                    os.path.join(OUT, a.shopper_ckpt), ()),
    }
    for name, (baseline, path, ignore) in specs.items():
        blob = strict_load(baseline, path, ignore=ignore)
        if name == "multinomial":
            with torch.no_grad():
                p = torch.as_tensor(law, dtype=baseline.log_pn.dtype)
                baseline.log_pn.copy_(torch.where(p > 0, torch.log(p),
                                                   torch.full_like(p, -float("inf"))))
        baseline.double().eval()
        records[name] = checkpoint_record(path, blob)

    result = dict(schema=1, created_unix=time.time(), seed=a.seed, nmax=a.nmax, R=a.R,
                  catalogue=J, partition=os.environ.get("V3_PARTITION", ""),
                  affinity=os.environ.get("V3_AFFINITY", "0"), manifest=manifest,
                  checkpoints=records, support={
                      "main": "nonempty; n<=nmax; per-category count<=R; complete assortment",
                      "multinomial": "same as main (exact restricted ESP denominator)",
                      "ndpp": "all nonempty subsets of complete assortment (conservative broader support)",
                      "shopper": "all nonempty sequences/sets of complete assortment (conservative broader support)"},
                  splits={})

    for si, (split, trips) in enumerate(picks.items()):
        log(f"scoring {split}: main")
        mv, lines = main_scores(model, main_batcher, trips, a.main_chunk)
        block = {"main": summarize(mv, lines)}
        raw = {"main": mv}
        for name, (baseline, _, _) in specs.items():
            log(f"scoring {split}: {name}")
            bv, blines = baseline_scores(
                baseline, baseline_batcher, trips, name, a.baseline_chunk,
                seed=a.seed + 1009 * si, shopper_orders=a.shopper_orders,
                exact_max_n=a.shopper_exact_max_n, category_cap=a.R)
            if not np.array_equal(lines, blines):
                raise RuntimeError(f"{name} line counts differ on the shared trip manifest")
            block[name] = summarize(bv, lines)
            block[name]["gap"] = paired(mv, bv)
            raw[name] = bv
        np.savez_compressed(os.path.join(OUT, f"{a.output}_{split}_per_trip.npz"),
                            trips=trips, lines=lines, **raw)
        result["splits"][split] = block

    out_path = os.path.join(OUT, f"{a.output}.json")
    with open(out_path, "w") as stream:
        json.dump(result, stream, indent=2)
    log("")
    for split, block in result["splits"].items():
        log(split.upper())
        for name, stats in block.items():
            gap = stats.get("gap", {}).get("main_minus_baseline")
            suffix = "" if gap is None else f"  main-gap {gap:+.3f}"
            log(f"  {name:12s} {stats['per_basket']:9.3f} +/- {stats['se_per_basket']:.3f}"
                f"  per-line {stats['per_line']:.4f}{suffix}")
    log(f"wrote {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--main-ckpt", default="v3_run112_blockscaled_safe_best.pt")
    parser.add_argument("--multinomial-ckpt", default="v3_bl_multinom.pt")
    parser.add_argument("--ndpp-ckpt", default="v3_bl_ndpp.pt")
    parser.add_argument("--shopper-ckpt", default="v3_bl_shopper.pt")
    parser.add_argument("--splits", default="valid,test")
    parser.add_argument("--n-trips", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--nmax", type=int, default=120)
    parser.add_argument("--R", type=int, default=23)
    parser.add_argument("--Kz", type=int, default=0)
    parser.add_argument("--main-chunk", type=int, default=24)
    parser.add_argument("--baseline-chunk", type=int, default=8)
    parser.add_argument("--shopper-orders", type=int, default=512)
    parser.add_argument("--shopper-exact-max-n", type=int, default=6)
    parser.add_argument("--output", default="v3_same_trips_verified")
    main(parser.parse_args())
