"""Train one audited basket baseline with resumable, provenance-complete checkpoints."""
import argparse
import hashlib
import json
import os
import time

import numpy as np
import torch

import evalall as EA
from baselines import Batches, Bernoulli, DPP
from baselines2 import Multinomial, NDPP, Shopper, size_law
from data import build
from features import Features


HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.abspath(os.path.join(HERE, "..", "..", "out"))


def log(message):
    print(f"[blv] {message}", flush=True)


def hash_ids(ids):
    return hashlib.sha256(np.ascontiguousarray(ids, dtype=np.int64).tobytes()).hexdigest()


def supported_training(D, nmax, R):
    ids = np.flatnonzero(D["trip_split"] == 0)
    lp, lc = D["line_ptr"], D["line_cat"]
    keep = []
    for t in ids:
        lo, hi = int(lp[t]), int(lp[t + 1])
        if 1 <= hi - lo <= nmax and np.bincount(lc[lo:hi]).max() <= R:
            keep.append(t)
    return np.asarray(keep, dtype=np.int64)


def absolute_popularity_logits(D, trips):
    """Training-only exposure-corrected log incidence, with its absolute level retained.

    RaggedModel centers this initializer because a common utility shift is a rho_0 gauge
    direction.  These external baselines have no rho_0, so that shift determines expected
    basket size and must not be removed.
    """
    J, C, S = (int(D[k]) for k in ("n_item", "n_cat", "n_store"))
    count = np.zeros(J, dtype=np.float64)
    for t in trips:
        lo, hi = int(D["line_ptr"][t]), int(D["line_ptr"][t + 1])
        count[D["line_item"][lo:hi]] += 1.0
    exposure = np.zeros(J, dtype=np.float64)
    store_n = np.bincount(D["trip_store"][trips], minlength=S)
    ptr, items = D["store_cat_ptr"], D["store_items"]
    for store in range(S):
        lo, hi = int(ptr[store * C]), int(ptr[(store + 1) * C])
        exposure[items[lo:hi]] += store_n[store]
    seen = exposure > 0
    value = np.empty(J, dtype=np.float64)
    value[seen] = np.log((count[seen] + 0.5) / (exposure[seen] + 1.0))
    value[~seen] = np.median(value[seen])
    return torch.as_tensor(value, dtype=torch.float64)


def make_model(name, D, a):
    J, N, S = (int(D[k]) for k in ("n_item", "n_user", "n_store"))
    if name == "multinomial":
        return Multinomial(J, N, S, size_law(D, a.nmax, a.R), K=a.K, Kp=a.Kp,
                           seed=a.seed, taste_init=a.taste_init)
    if name == "bernoulli":
        return Bernoulli(J, N, S, K=a.K, Kp=a.Kp, seed=a.seed,
                         taste_init=a.taste_init)
    if name == "dpp":
        return DPP(J, N, S, rank=a.rank, K=a.K, Kp=a.Kp, seed=a.seed,
                   taste_init=a.taste_init, interaction_init=a.interaction_init)
    if name == "ndpp":
        return NDPP(J, N, S, rank=a.rank, srank=a.srank, K=a.K, Kp=a.Kp,
                    seed=a.seed, taste_init=a.taste_init,
                    interaction_init=a.interaction_init)
    if name == "shopper":
        return Shopper(J, N, S, K=a.K, Kp=a.Kp, Ki=a.interaction_rank,
                       seed=a.seed, taste_init=a.taste_init,
                       interaction_init=a.interaction_init)
    raise ValueError(name)


def call(model, name, batch, a, generator, evaluation=False):
    if name == "multinomial":
        return model.loglik(batch, category_cap=a.R)
    if name == "bernoulli":
        return model.loglik(batch, nmax=a.nmax, category_cap=a.R)
    if name == "shopper":
        return model.loglik(batch,
                            n_orders=a.eval_orders if evaluation else a.train_orders,
                            exact_max_n=a.exact_max_n if evaluation else 0,
                            gen=generator, max_size=a.nmax)
    return model.loglik(batch)


@torch.no_grad()
def evaluate(model, name, batcher, trips, a, seed):
    values, lines = [], 0
    generator = torch.Generator().manual_seed(seed)
    for k in range(0, len(trips), a.eval_chunk):
        sub = trips[k:k + a.eval_chunk]
        d = batcher.make(sub)
        ll = call(model, name, d, a, generator, evaluation=True)
        if not torch.isfinite(ll).all():
            raise RuntimeError(f"non-finite {name} validation score")
        values.extend(ll.tolist())
        lines += len(d["li"])
    x = np.asarray(values)
    return dict(per_basket=float(x.mean()),
                se=float(x.std(ddof=1) / np.sqrt(len(x))),
                per_line=float(x.sum() / lines), n=len(x))


def save(path, model, opt, sched, iteration, best, a, name, tr, va, te,
         rng, order_gen):
    blob = dict(format=3, kind="verified-basket-baseline", model_name=name,
                model=model.state_dict(), optimizer=opt.state_dict(),
                scheduler=sched.state_dict(), iteration=iteration, best=best,
                config=vars(a), data=dict(partition=os.environ.get("V3_PARTITION", ""),
                                           affinity=os.environ.get("V3_AFFINITY", "0"),
                                           n_item=int(model.idx.lam.shape[0]),
                                           train_n=len(tr), train_hash=hash_ids(tr),
                                           valid_n=len(va), valid_hash=hash_ids(va),
                                           test_n=len(te), test_hash=hash_ids(te)),
                rng_np=rng.bit_generator.state, rng_torch=order_gen.get_state())
    tmp = path + ".tmp"
    torch.save(blob, tmp)
    os.replace(tmp, path)


def main(a):
    torch.set_flush_denormal(True)
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(a.seed)
    D = build()
    J, S = int(D["n_item"]), int(D["n_store"])
    batcher = Batches(D, Features(J, S, 712))
    tr = supported_training(D, a.nmax, a.R)
    # Exact fit.py manifest: support filter, fixed permutation, then the prefix.  A raw
    # prefix is about eight nats easier on these data; an independently sampled manifest
    # is legitimate for a separate experiment but not for the requested matched audit.
    def fit_manifest(split_code, seed):
        ids = np.flatnonzero(D["trip_split"] == split_code)
        lp, lc = D["line_ptr"], D["line_cat"]
        keep = []
        for t in ids:
            lo, hi = int(lp[t]), int(lp[t + 1])
            if 1 <= hi - lo <= a.nmax and np.bincount(lc[lo:hi]).max() <= a.R:
                keep.append(t)
        ids = np.asarray(keep, dtype=np.int64)
        ids = ids[np.random.default_rng(seed).permutation(len(ids))]
        return ids[:a.n_val]
    va = fit_manifest(1, a.manifest_seed)
    te = fit_manifest(2, a.manifest_seed)
    log(f"{a.model}: {len(tr):,} training trips ({hash_ids(tr)[:12]}); "
        f"valid {hash_ids(va)[:12]}, test {hash_ids(te)[:12]}")
    model = make_model(a.model, D, a).double()
    if a.init_popularity:
        with torch.no_grad():
            model.idx.lam.copy_(absolute_popularity_logits(D, tr))
        log(f"{a.model}: initialized product intercept from training incidence/exposure")
    opt = torch.optim.Adam(model.parameters(), lr=a.lr, weight_decay=a.wd)
    milestones = [int(x) for x in a.lr_milestones.split(",") if x.strip()]
    if milestones:
        sched = torch.optim.lr_scheduler.MultiStepLR(
            opt, milestones=milestones, gamma=a.lr_gamma)
    else:
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=a.iters, eta_min=a.lr * a.lr_floor)
    rng = np.random.default_rng(a.seed)
    order_gen = torch.Generator().manual_seed(a.seed)
    suffix = f"_{a.tag}" if a.tag else ""
    path = os.path.join(OUT, f"baseline_verified_{a.model}{suffix}.pt")
    best_path = os.path.join(OUT, f"baseline_verified_{a.model}{suffix}_best.pt")
    start, best = 0, dict(per_basket=-float("inf"), iteration=0)
    if a.resume and os.path.exists(path):
        blob = torch.load(path, map_location="cpu", weights_only=False)
        if blob.get("format") != 3 or blob.get("model_name") != a.model:
            raise RuntimeError("resume checkpoint is not the requested verified baseline")
        model.load_state_dict(blob["model"], strict=True)
        opt.load_state_dict(blob["optimizer"]); sched.load_state_dict(blob["scheduler"])
        start, best = int(blob["iteration"]), blob["best"]
        rng.bit_generator.state = blob["rng_np"]; order_gen.set_state(blob["rng_torch"])
        log(f"resumed iteration {start}, best {best['per_basket']:.4f}")
    if a.eval_initial:
        initial = evaluate(model.eval(), a.model, batcher, va, a, a.eval_seed)
        model.train()
        log(f"initial it {start:6d}: valid {initial['per_basket']:.3f} "
            f"+/- {initial['se']:.3f} per-line {initial['per_line']:.4f}")
    started = time.time()
    for iteration in range(start + 1, a.iters + 1):
        trips = tr[rng.choice(len(tr), size=a.batch, replace=False)]
        d = batcher.make(trips)
        loss = -call(model, a.model, d, a, order_gen).mean()
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at iteration {iteration}")
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip)
        opt.step(); sched.step()
        if iteration % a.eval_every == 0 or iteration == a.iters:
            model.eval()
            val = evaluate(model, a.model, batcher, va, a, a.eval_seed)
            model.train()
            elapsed = time.time() - started
            log(f"it {iteration:6d}/{a.iters} loss {float(loss.detach()):9.3f} "
                f"valid {val['per_basket']:9.3f} +/- {val['se']:.3f} "
                f"lr {opt.param_groups[0]['lr']:.2g} {elapsed/60:.1f} min "
                f"ETA {(elapsed/max(iteration-start,1))*(a.iters-iteration)/3600:.2f} h")
            if val["per_basket"] > best["per_basket"]:
                best = dict(per_basket=val["per_basket"], iteration=iteration)
                save(best_path, model, opt, sched, iteration, best, a, a.model,
                     tr, va, te, rng, order_gen)
            save(path, model, opt, sched, iteration, best, a, a.model,
                 tr, va, te, rng, order_gen)
    best_blob = torch.load(best_path, map_location="cpu", weights_only=False)
    model.load_state_dict(best_blob["model"], strict=True); model.eval()
    valid = evaluate(model, a.model, batcher, va, a, a.eval_seed)
    test = evaluate(model, a.model, batcher, te, a, a.eval_seed + 1009)
    result = dict(model=a.model, best=best_blob["best"], valid=valid, test=test,
                  checkpoint=os.path.basename(best_path), config=vars(a))
    with open(os.path.join(OUT, f"baseline_verified_{a.model}{suffix}.json"), "w") as stream:
        json.dump(result, stream, indent=2)
    log(f"best iteration {best['iteration']}: valid {valid['per_basket']:.4f}, "
        f"test {test['per_basket']:.4f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["multinomial", "bernoulli", "dpp", "ndpp", "shopper"], required=True)
    p.add_argument("--tag", default="")
    p.add_argument("--iters", type=int, default=60000)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--lr-floor", type=float, default=0.02)
    p.add_argument("--lr-milestones", default="20000,26000")
    p.add_argument("--lr-gamma", type=float, default=0.5)
    p.add_argument("--wd", type=float, default=1e-5)
    p.add_argument("--clip", type=float, default=5.0)
    p.add_argument("--eval-every", type=int, default=2000)
    p.add_argument("--n-val", type=int, default=512)
    p.add_argument("--eval-chunk", type=int, default=8)
    p.add_argument("--eval-initial", type=int, default=1,
                   help="score the raw/resumed checkpoint before any updates")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--taste-init", type=float, default=0.03)
    p.add_argument("--interaction-init", type=float, default=0.03)
    p.add_argument("--init-popularity", type=int, default=1)
    p.add_argument("--eval-seed", type=int, default=20260821)
    p.add_argument("--manifest-seed", type=int, default=12345)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--nmax", type=int, default=120)
    p.add_argument("--R", type=int, default=23)
    p.add_argument("--K", type=int, default=32)
    p.add_argument("--Kp", type=int, default=8)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--srank", type=int, default=8)
    p.add_argument("--interaction-rank", type=int, default=16)
    p.add_argument("--train-orders", type=int, default=4)
    p.add_argument("--eval-orders", type=int, default=128)
    p.add_argument("--exact-max-n", type=int, default=6)
    main(p.parse_args())
