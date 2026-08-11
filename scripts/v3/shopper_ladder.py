"""Ordering ladder for SHOPPER, from a saved checkpoint.

SHOPPER's likelihood is over ORDERED baskets, so a set probability needs
P(S) = n! E_pi[P(pi)] over sampled orderings.  That estimator is unbiased for P(S) and so
biased LOW for log P(S); the bias falls with the number of orderings, and the only way to
know whether a quoted number has converged is to walk the ladder out until it stops moving.

Run from the saved model so the ladder costs no training, and start small: the cost is
linear in orderings times trips, and 2048 orderings on 512 trips is 21x the work of the
first ladder I ran, which is hours rather than minutes.
"""
import argparse, json, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from baselines import Batches
from baselines2 import Shopper, ev
from data import build
from features import Features

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "out")


def main(a):
    torch.set_default_dtype(torch.float64)
    D = build()
    J, N, S = (int(D[k]) for k in ("n_item", "n_user", "n_store"))
    F = Features(J, S, 712)
    Bt = Batches(D, F)
    va = np.flatnonzero(D["trip_split"] == 1)
    m = Shopper(J, N, S, K=a.K, Kp=a.Kp)
    m.load_state_dict(torch.load(os.path.join(OUT, "v3_shopper.pt")))
    print(f"[lad] loaded out/v3_shopper.pt; ladder on {a.n_val} validation trips",
          flush=True)
    res, prev = {}, None
    for no in a.ladder:
        t0 = time.time()
        vb, vl = ev(m, Bt, va[:a.n_val], n_orders=no)
        step = "" if prev is None else f"   step {vb - prev:+.4f}"
        print(f"[lad] {no:5d} orderings: {vb:9.4f} / basket   {vl:8.4f} / line"
              f"   {(time.time()-t0)/60:5.1f} min{step}", flush=True)
        res[str(no)] = vb
        prev = vb
        json.dump(res, open(os.path.join(OUT, "v3_shopper_ladder.json"), "w"), indent=2)
    print("[lad] wrote out/v3_shopper_ladder.json", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--K", type=int, default=32)
    p.add_argument("--Kp", type=int, default=8)
    p.add_argument("--n-val", type=int, default=512)
    p.add_argument("--ladder", type=int, nargs="*", default=[8, 32, 128, 512])
    main(p.parse_args())
