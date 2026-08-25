"""Fair, reproducible version-4 interaction-versus-multinomial audit.

Here "multinomial" means the strict nested version-4 model: the *same* joint law,
utility parameterisation, size potential, support, covariates, and unit likelihood, with
only ``phi`` and ``rho_c`` fixed to zero.  This is the only comparison that isolates the
value of the version-4 interaction terms.  The externally factored BEMB adaptation in
``baselines2.py`` uses an empirical, context-free P(n), so it answers a different question.

The script refuses resumed checkpoints, mismatched update counts/configurations, partial
support, non-affinity partitions, or a nonzero interaction in the nested checkpoint.  It
scores the first N validation trips used by fit.py, saves their exact ordered IDs and hash,
and reports paired per-trip gaps for joint set, size, composition, and unit likelihood.
"""
import argparse
import hashlib
import json
import os
import time

import numpy as np
import torch

from bench_same_trips import OUT, checkpoint_record, load_main, paired, summarize
from data import build
from decompose_same_trips import score_main
from features import Features
from fit import Batcher


def sha_ids(ids):
    raw = np.ascontiguousarray(ids, dtype=np.int64).tobytes()
    return hashlib.sha256(raw).hexdigest()


def load_json(path):
    with open(path) as stream:
        return json.load(stream)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def config_of(blob):
    return blob.get("config", blob)


def audit_configs(full, nested):
    cf, cn = config_of(full), config_of(nested)
    required_equal = (
        "K", "Kp", "Kz", "R", "nmax", "batch", "lr", "wd", "seed",
        "no_rec", "units", "init_popularity", "init_rho0", "taste_init",
        "lam_lr_scale", "lam_centre", "lam_project", "lam_sd_max",
        "pool_prod", "en_w", "rkl_w", "size_kl", "beta_cal_w",
        "elast_w", "elast_target", "n_val",
    )
    mismatches = {key: [cf.get(key), cn.get(key)] for key in required_equal
                  if cf.get(key) != cn.get(key)}
    require(not mismatches, f"training configurations differ: {mismatches}")
    require(not cf.get("resume") and not cf.get("warm_start"),
            "full model is not a fresh-start experiment")
    require(not cn.get("resume") and not cn.get("warm_start"),
            "nested multinomial is not a fresh-start experiment")
    require(int(cf.get("zero_phi", 0)) == 0 and int(cf.get("zero_rho_c", 0)) == 0,
            "full checkpoint config disables an interaction block")
    require(int(cn.get("zero_phi", 0)) == 1 and int(cn.get("zero_rho_c", 0)) == 1,
            "nested checkpoint config did not explicitly freeze both interactions")
    require(int(cf["R"]) == int(cf["nmax"]) == 120,
            "comparison is not on version-4 complete support R=nmax=120")
    return {key: cf[key] for key in required_equal}


def checkpoint_meta(path, blob):
    rec = checkpoint_record(path, blob)
    rec["iteration"] = int(blob.get("iter", blob.get("iteration", -1)))
    rec["data"] = blob.get("data", {})
    return rec


def main(args):
    torch.set_flush_denormal(True)
    torch.set_default_dtype(torch.float64)
    require(os.environ.get("V3_AFFINITY") == "1",
            "set V3_AFFINITY=1; version-4 uses the 280-row affinity partition")
    data = build()
    require(int(data["n_item"]) == 5455 and int(data["n_cat"]) == 280,
            "data are not the 5,455-product, 280-affinity version-4 universe")

    full_path = os.path.join(OUT, args.full_ckpt)
    nested_path = os.path.join(OUT, args.multinomial_ckpt)
    full_blob = torch.load(full_path, map_location="cpu", weights_only=False)
    nested_blob = torch.load(nested_path, map_location="cpu", weights_only=False)
    require(int(full_blob.get("iter", -1)) == int(nested_blob.get("iter", -2)),
            "checkpoints are not matched at the same optimizer-update count")
    for name, blob in (("full", full_blob), ("multinomial", nested_blob)):
        md = blob.get("data", {})
        require(md.get("affinity") == "1" and int(md.get("n_cat", -1)) == 280,
                f"{name} checkpoint was not trained on the affinity partition")
        require(int(md.get("n_item", -1)) == 5455,
                f"{name} checkpoint does not cover all products")
        require(int(md.get("R", -1)) == int(md.get("nmax", -2)) == 120,
                f"{name} checkpoint does not use complete support")

    full_cfg_path = os.path.join(OUT, args.full_config)
    nested_cfg_path = os.path.join(OUT, args.multinomial_config)
    matched_config = audit_configs(load_json(full_cfg_path), load_json(nested_cfg_path))

    sd_nested = nested_blob["model"]
    require(torch.count_nonzero(sd_nested["phi"]).item() == 0,
            "nested multinomial has nonzero latent interactions")
    require(torch.count_nonzero(sd_nested["rho_c"]).item() == 0,
            "nested multinomial has nonzero category-count interactions")

    all_valid = np.flatnonzero(data["trip_split"] == 1).astype(np.int64)
    # This is deliberately byte-for-byte fit.py's validation construction.  The trainer
    # first filters support (a no-op for R=nmax=120 here), then applies this fixed
    # permutation before taking va[:n_val].  Using the raw prefix is a materially easier
    # slice and changed the score by roughly eight nats in the first audit attempt.
    valid_order = np.random.default_rng(12345).permutation(len(all_valid))
    trips = all_valid[valid_order][:args.n_val]
    require(len(trips) == args.n_val, "validation split is smaller than requested audit")
    ptr = data["line_ptr"]
    sizes = ptr[trips + 1] - ptr[trips]
    require(bool(np.all((sizes >= 1) & (sizes <= 120))),
            "the fit.py validation manifest contains a basket outside support")

    features = Features(int(data["n_item"]), int(data["n_store"]), 712)
    batcher = Batcher(data, features, 120)
    full, _, full_load_meta = load_main(full_path, data, 32, 120, 120)
    nested, _, nested_load_meta = load_main(nested_path, data, 32, 120, 120)

    print(f"[fair] validation {len(trips)} trips, hash {sha_ids(trips)[:16]}", flush=True)
    print("[fair] scoring full interaction model", flush=True)
    sf = score_main(full, batcher, trips, args.chunk)
    print("[fair] scoring strict nested multinomial", flush=True)
    sn = score_main(nested, batcher, trips, args.chunk)
    require(np.array_equal(sf["lines"], sn["lines"]), "basket line counts differ")

    # The unit model is outside the set-interaction comparison, but score it on the same
    # baskets to expose any accidental change in the target or batching contract.
    def unit_scores(model):
        values = []
        for k in range(0, len(trips), args.chunk):
            sub = trips[k:k + args.chunk]
            ix, ctx, lctx, hh, li, lt, _, lq = batcher.make(sub)
            model.house, model.ctx = hh, ctx
            with torch.no_grad():
                values.extend(model.units_loglik(li, lt, lq, lctx, ix.B).tolist())
        return np.asarray(values)

    sf["units"] = unit_scores(full)
    sn["units"] = unit_scores(nested)

    sections = {}
    for part in ("joint", "size", "composition", "units"):
        sections[part] = {
            "full": summarize(sf[part], sf["lines"]),
            "multinomial": summarize(sn[part], sn["lines"]),
            "paired_full_minus_multinomial": paired(sf[part], sn[part]),
        }
        gap = sections[part]["paired_full_minus_multinomial"]
        print(f"[fair] {part:11s} full {sf[part].mean():9.5f}  "
              f"multi {sn[part].mean():9.5f}  gap {gap['main_minus_baseline']:+.5f} "
              f"+/- {gap['paired_se']:.5f}", flush=True)

    train = np.flatnonzero(data["trip_split"] == 0).astype(np.int64)
    result = {
        "schema": 1,
        "created_unix": time.time(),
        "definition": ("strict nested version-4 multinomial: identical original joint "
                       "law with phi=rho_c=0, not externally factored empirical P(n)"),
        "fairness": {
            "fresh_start": True,
            "matched_updates": int(full_blob["iter"]),
            "matched_config": matched_config,
            "same_training_split_hash": sha_ids(train),
            "same_ordered_validation_ids": True,
            "complete_support": "all 5,455 products; 1<=n<=120; R=120",
            "partition": "280 affinity rows",
            "score_used_for_claim": "joint set log likelihood per basket",
            "units_excluded_from_set_claim": True,
        },
        "validation_manifest": {
            "selection": ("fit.py validation order: all in-support validation IDs, "
                          "permuted by numpy default_rng(12345), then va[:n_val]"),
            "permutation_seed": 12345,
            "n": len(trips),
            "sha256": sha_ids(trips),
            "first_ids": trips[:20].tolist(),
            "line_count": int(sizes.sum()),
            "min_size": int(sizes.min()),
            "max_size": int(sizes.max()),
        },
        "checkpoints": {
            "full": checkpoint_meta(full_path, full_blob),
            "multinomial": checkpoint_meta(nested_path, nested_blob),
        },
        "loader_meta": {"full": full_load_meta, "multinomial": nested_load_meta},
        "scores": sections,
    }
    stem = os.path.join(OUT, args.output)
    np.savez_compressed(stem + "_per_trip.npz", trips=trips, lines=sf["lines"],
                        full_joint=sf["joint"], multinomial_joint=sn["joint"],
                        full_size=sf["size"], multinomial_size=sn["size"],
                        full_composition=sf["composition"],
                        multinomial_composition=sn["composition"],
                        full_units=sf["units"], multinomial_units=sn["units"])
    with open(stem + ".json", "w") as stream:
        json.dump(result, stream, indent=2)
    print(f"[fair] wrote {stem}.json and {stem}_per_trip.npz", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-ckpt", required=True)
    parser.add_argument("--multinomial-ckpt", required=True)
    parser.add_argument("--full-config", required=True)
    parser.add_argument("--multinomial-config", required=True)
    parser.add_argument("--n-val", type=int, default=384)
    parser.add_argument("--chunk", type=int, default=24)
    parser.add_argument("--output", default="v3_version4_multinomial_fair_audit")
    main(parser.parse_args())
