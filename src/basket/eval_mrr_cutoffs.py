"""Evaluate complete-the-basket MRR and truncated MRR@K on fit.py's fixed holdouts.

MRR@K is zero when the held-out item's rank exceeds K.  Recall@K is reported beside it so
the two commonly-confused quantities remain explicit.
"""
import argparse
import os

import numpy as np
import torch

import ragged
from data import build
from diagnose_bucket_coverage import legacy_esp_bucketed
from evalall import load_any
from features import Features
from fit import Batcher, rec_eval
from ragged import RaggedModel


_HERE = os.path.dirname(os.path.abspath(__file__))


def _resolve_ckpt(p):
    """Accept an absolute path, a path relative to the CWD, or a bare checkpoint name.

    Blindly prefixing "../../out" turned a perfectly good relative path into
    ../../out/../../out/<name>.  Take the path as given when it exists; only fall back to
    the repository's out/ directory for a bare name.
    """
    if os.path.exists(p):
        return p
    cand = os.path.join(_HERE, "..", "..", "out", os.path.basename(p))
    if os.path.exists(cand):
        return cand
    raise SystemExit(f"checkpoint not found: {p} (also tried {os.path.normpath(cand)})")



def main(a):
    torch.set_default_dtype(torch.float64)
    D = build()
    J, N, C, S = (int(D[k]) for k in ("n_item", "n_user", "n_cat", "n_store"))
    path = _resolve_ckpt(a.ckpt)
    blob = torch.load(path, map_location="cpu", weights_only=False)
    q = blob.get("quad") or {}
    kz = int(q.get("Kz", blob["model"]["phi"].shape[1]))
    model = RaggedModel(J=J, N=N, C=C, K=32, Kz=kz, nmax=a.nmax, R=a.R, S=S, Kp=8)
    meta = load_any(path, model, J, D)
    model.double().eval()
    # The price reference is a property of the CHECKPOINT.  Scoring a
    # category-referenced model against a trip-wide mean deletes its
    # substitution channel and changes every score.
    _ref = str((blob.get("model_flags") or {}).get("price_ref", "trip")) \
           if isinstance(blob, dict) else "trip"
    batcher = Batcher(D, Features(J, S, 712), a.nmax, price_ref=_ref)
    if a.legacy_buckets:
        ragged.esp_bucketed = legacy_esp_bucketed

    trips = np.flatnonzero(D["trip_split"] == 1)
    lp, lc = D["line_ptr"], D["line_cat"]
    keep = []
    for t in trips:
        lo, hi = int(lp[t]), int(lp[t + 1])
        if hi - lo <= a.nmax and (hi <= lo or np.bincount(lc[lo:hi]).max() <= a.R):
            keep.append(t)
    trips = np.asarray(keep, dtype=int)
    trips = trips[np.random.default_rng(12345).permutation(len(trips))][:a.n_trips]
    ranks = rec_eval(model, batcher, trips, seed=0, chunk=a.chunk, return_ranks=True)
    print(f"checkpoint: {os.path.basename(path)} ({meta})")
    print(f"cases: {len(ranks)}")
    print(f"MRR: {np.mean(1.0 / ranks):.6f}")
    print(f"median rank: {np.median(ranks):.0f}")
    for k in a.cutoffs:
        hit = ranks <= k
        print(f"MRR@{k}: {np.mean(np.where(hit, 1.0 / ranks, 0.0)):.6f}  "
              f"R@{k}: {np.mean(hit):.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="v3_run112_blockscaled_safe_bestmrr.pt")
    parser.add_argument("--n-trips", type=int, default=192)
    parser.add_argument("--chunk", type=int, default=24)
    parser.add_argument("--nmax", type=int, default=120)
    parser.add_argument("--R", type=int, default=23)
    parser.add_argument("--cutoffs", type=int, nargs="+", default=[5, 10, 20])
    parser.add_argument("--legacy-buckets", action="store_true")
    main(parser.parse_args())
