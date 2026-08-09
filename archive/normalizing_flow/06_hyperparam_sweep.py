"""
Stage 6 -- Hyperparameter search for the product-choice stage.

The paper (app. 8.2.1) grid-searches K, the price dimension, whether to include
user/item observables, the learning rate and linear-vs-log price, and -- crucially
-- selects on *counterfactual* performance: validation log-likelihood restricted to
item-weeks in which the price actually moved, not on the validation set at large.
The same criterion is used here.
"""
import argparse
import itertools
import json
import os
import time

import numpy as np
import pandas as pd
import torch

import nf_torch as nf
from importlib import import_module

trainer = import_module("05_train_nf")
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "..", "out")


class Cfg:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def main(a):
    dev = trainer.pick_device(a.device)
    os.makedirs(OUT, exist_ok=True)
    d = nf.load(trainer.MI, device=dev)
    rows = []
    grid = list(itertools.product(a.K, a.Kp, a.price_prior_var, a.lr))
    print(f"[06] device {dev};  {len(grid)} configurations", flush=True)
    for k, kp, ppv, lr in grid:
        # Every attribute train_stage1 reads must be present; a missing one is an
        # AttributeError 1,500 iterations in.  Keep this in step with 05_train_nf.py.
        cfg = Cfg(K=k, Kp=kp, price_prior_var=ppv, lr=lr, iters=a.iters, batch=5000,
                  eval_every=a.eval_every, label=f"sweep_K{k}_Kp{kp}_pv{ppv}_lr{lr}",
                  no_user_obs=a.no_user_obs, item_obs=a.item_obs, homogeneous=False,
                  extras=[], no_pool=False, price_prior_mean=a.price_prior_mean,
                  prior_var=1.0, intercept_var=10.0, no_scale_prior=False, seed=a.seed)
        t0 = time.time()
        _, hist = trainer.train_stage1(d, cfg, dev)
        best = max(h["val_loglik_price_weeks"] for h in hist)
        best_all = max(h["val_loglik"] for h in hist)
        rows.append({"K": k, "Kp": kp, "price_prior_var": ppv, "lr": lr,
                     "val_ll_price_weeks": best, "val_ll_all": best_all,
                     "secs": time.time() - t0})
        print(f"[06] K={k:3d} Kp={kp:3d} priceVar={ppv:5.2f} lr={lr:g}  "
              f"val(price wks) {best:.4f}  val {best_all:.4f}  "
              f"({time.time()-t0:.0f}s)", flush=True)
        os.remove(os.path.join(OUT, f"{cfg.label}_stage1.pt"))
    df = pd.DataFrame(rows).sort_values("val_ll_price_weeks", ascending=False)
    df.to_csv(os.path.join(OUT, "sweep_stage1.csv"), index=False)
    print("\n" + df.to_string(index=False))
    print(f"\n[06] selected: {df.iloc[0].to_dict()}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--K", type=int, nargs="+", default=[10, 20, 40, 80])
    p.add_argument("--Kp", type=int, nargs="+", default=[5, 10, 20])
    p.add_argument("--price-prior-var", type=float, nargs="+", default=[0.25, 1.0])
    p.add_argument("--lr", type=float, nargs="+", default=[0.005])
    p.add_argument("--price-prior-mean", type=float, default=0.5)
    p.add_argument("--iters", type=int, default=1500)
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--no-user-obs", action="store_true")
    p.add_argument("--item-obs", action="store_true")
    # cpu, not auto: on Apple silicon "auto" selects MPS, whose float32
    # reductions differ enough to move the reported log-likelihoods.  Pass
    # --device mps or --device cuda explicitly to opt in.
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=0)
    main(p.parse_args())
