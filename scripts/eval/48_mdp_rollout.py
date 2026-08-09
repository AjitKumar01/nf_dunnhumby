"""
Stage 48 -- Closed-loop trajectory simulation, and what it costs.

Every generation result in this repository so far is OPEN LOOP: `generate_baskets` reads
the recency state from the household's REAL purchase history, so a generated novel product
never makes the next generated basket more novel.  Section 11 of the experiments page says
so and calls its drift measurement a lower bound.  This stage closes the loop and measures
how much of a lower bound it was.

WHAT THE MODEL CARRIES AS STATE.  Four things enter utility from a household's past:

    d.state(u, j, t)      days since u last bought j's SUB-COMMODITY   dynamic
    d.cat_state(u, c, t)  days since u last bought anything in c       dynamic
    d.loyal / d.freq      u's within-category share and log-count of j FROZEN at train
    d.hh_cat              u's log-count of purchases in c              FROZEN at train

The first two are recomputed per trip and can be fed back.  The last two are fixed
lookups built once from the training window, so in a rollout a household that starts
buying something new NEVER becomes loyal to it: the habit terms are stuck at their
training values however long the trajectory runs.  That is a structural ceiling on
horizon, independent of any fit quality, and this stage measures its size by running the
rollout with and without those terms updated too.

THREE MODES, each a strictly larger feedback loop:

    open        recency from the real history            what the repo does today
    recency     recency from the simulated history       true MDP transition on x
    full        recency + habit counts updated           what a policy would actually see

Trip TIMING is held at the household's real trip days in all three.  The model conditions
on the day and never generates it, so inter-trip gaps are exogenous; simulating them
needs a hazard model this work does not have.  Stated rather than hidden.

Writes out/mdp_rollout_<label>.json.
"""
import argparse
import importlib
import json
import os
import sys
import time

import numpy as np
import torch

# the model lives in ../model; add it to the path so `27_nested_basket` and
# `28_nested_counterfactual` resolve by their bare module names.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "model"))
nb = importlib.import_module("27_nested_basket")
cf = importlib.import_module("28_nested_counterfactual")

HERE = os.path.dirname(os.path.abspath(__file__))
IN = os.path.join(HERE, "..", "..", "basket_input")
OUT = os.path.join(HERE, "..", "..", "out")


def log(m):
    print(f"[48] {m}", flush=True)


class SimHistory:
    """A household's purchase history as the simulation believes it to be.

    Seeded from the real history at t0 so the trajectory starts where the household
    actually is, then advanced only by what the model generates.
    """

    def __init__(self, d, users, t0, update_habit):
        self.d, self.update_habit = d, update_habit
        self.users = np.asarray(users)
        self.uix = {int(u): k for k, u in enumerate(self.users)}
        H = len(self.users)
        self.last_sub = np.full((H, d.S), -1, dtype=np.int64)
        self.last_cat = np.full((H, d.C), -1, dtype=np.int64)

        # seed both from the real arrays, exactly as d.state would read them at t0
        for k, u in enumerate(self.users):
            g = int(u) * d.S + np.arange(d.S)
            idx = np.searchsorted(d.keys, g * d.stride + t0, side="left") - 1
            ok = idx >= 0
            pk = d.keys[np.clip(idx, 0, len(d.keys) - 1)]
            hit = ok & ((pk // d.stride) == g)
            self.last_sub[k, hit] = pk[hit] % d.stride
            gc = int(u) * d.C + np.arange(d.C)
            idc = np.searchsorted(d.cat_keys, gc * d.stride + t0, side="left") - 1
            okc = idc >= 0
            pc = d.cat_keys[np.clip(idc, 0, len(d.cat_keys) - 1)]
            hc = okc & ((pc // d.stride) == gc)
            self.last_cat[k, hc] = pc[hc] % d.stride

        # habit counts, seeded from the frozen training tensors
        self.cnt_j = (np.expm1(d.freq.numpy()[self.users] * np.log(100.0))).astype(np.float64)
        self.cnt_c = (np.expm1(d.hh_cat.numpy()[self.users] * np.log(100.0))).astype(np.float64)

    # --- the two dynamic features, recomputed from the simulated history -------------
    def _feat(self, since, same, gap):
        return np.stack([
            (~same).astype(np.float32),
            np.where(same, np.exp(-since / 7.0), 0.0),
            np.where(same, np.exp(-since / gap), 0.0),
            np.where(same, np.log1p(since) / np.log(100.0), 0.0)],
            axis=1).astype(np.float32)

    def state(self, user, item, day):
        u = np.array([self.uix[int(x)] for x in np.asarray(user)])
        sub = self.d.item_sub[np.asarray(item)]
        last = self.last_sub[u, sub]
        same = last >= 0
        since = np.where(same, np.asarray(day) - last, 0).astype(np.float32)
        return self._feat(since, same, self.d.sub_gap[sub])

    def cat_state(self, user, cat, day):
        u = np.array([self.uix[int(x)] for x in np.asarray(user)])
        c = np.asarray(cat)
        last = self.last_cat[u, c]
        same = last >= 0
        since = np.where(same, np.asarray(day) - last, 0).astype(np.float32)
        return self._feat(since, same, self.d.cat_gap[c])

    def record(self, user, items, day):
        if len(items) == 0:
            return
        k = self.uix[int(user)]
        items = np.asarray(items)
        self.last_sub[k, self.d.item_sub[items]] = day
        self.last_cat[k, self.d.item_cat_np[items]] = day
        if self.update_habit:
            np.add.at(self.cnt_j[k], items, 1.0)
            np.add.at(self.cnt_c[k], self.d.item_cat_np[items], 1.0)

    def push_habit(self, d):
        """Write the simulated counts back into the tensors utility reads."""
        if not self.update_habit:
            return
        f = (np.log1p(self.cnt_j) / np.log(100.0)).astype(np.float32)
        d.freq[torch.as_tensor(self.users)] = torch.as_tensor(f)
        hc = (np.log1p(self.cnt_c) / np.log(100.0)).astype(np.float32)
        d.hh_cat[torch.as_tensor(self.users)] = torch.as_tensor(hc)
        cc = self.cnt_c[:, self.d.item_cat_np]
        d.loyal[torch.as_tensor(self.users)] = torch.as_tensor(
            (self.cnt_j / np.maximum(cc, 1.0)).astype(np.float32))


def rollout(m, d, dev, hh_trips, mode, seed, sweeps, item_temp=1.0,
            require_nonempty=False):
    """Roll every household forward together, one trip index at a time."""
    users = [u for u, _ in hh_trips]
    T = min(len(t) for _, t in hh_trips)
    sp = d.splits["test"]
    t0 = int(min(sp["day"][sp["starts"][t[0]]] for _, t in hh_trips))

    real_state, real_cat = d.state, d.cat_state
    real_freq = d.freq.clone(); real_hh = d.hh_cat.clone(); real_loyal = d.loyal.clone()
    sim = None
    if mode != "open":
        sim = SimHistory(d, users, t0, update_habit=(mode == "full"))
        d.state, d.cat_state = sim.state, sim.cat_state

    per_step, t_start = [], time.time()
    try:
        for step in range(T):
            trips = np.array([t[step] for _, t in hh_trips])
            g = cf.generate_baskets(m, d, dev, n_trips=len(trips), seed=seed + step,
                                    sweeps=sweeps, use_ctx=True, with_units=False,
                                    trips=trips, item_temp=item_temp,
                                    require_nonempty=require_nonempty)
            sizes = [len(b) for b in g]
            if sim is not None:
                for b, tr in zip(g, g.trips):
                    u = int(sp["user"][sp["starts"][tr]])
                    day = int(sp["day"][sp["starts"][tr]])
                    sim.record(u, np.asarray(b), day)
                sim.push_habit(d)
            per_step.append({"step": step, "mean_items": float(np.mean(sizes)),
                             "trips": [int(x) for x in g.trips],
                             "baskets": [np.asarray(b).tolist() for b in g]})
    finally:
        d.state, d.cat_state = real_state, real_cat
        d.freq.copy_(real_freq); d.hh_cat.copy_(real_hh); d.loyal.copy_(real_loyal)
    return per_step, time.time() - t_start


def novelty_curve(steps, d, seen0):
    """Cumulative distinct never-before-bought products per household, by step."""
    sp = d.splits["test"]
    acc, out = {}, []
    for st in steps:
        for tr, b in zip(st["trips"], st["baskets"]):
            u = int(sp["user"][sp["starts"][tr]])
            a = acc.setdefault(u, set())
            for j in b:
                if not seen0[u, int(j)]:
                    a.add(int(j))
        out.append(float(np.mean([len(acc.get(u, ())) for u in acc])) if acc else 0.0)
    return out


def main(a):
    dev = torch.device("cpu")
    d = nb.NestedData(IN, device=dev)
    m, _ = cf.load(a.label, d, dev)
    sp, tr = d.splits["test"], d.splits["train"]

    seen0 = np.zeros((d.N, d.J), dtype=bool)
    seen0[tr["user"], tr["item"]] = True

    by_hh = {}
    for i in range(sp["n_baskets"]):
        u = int(sp["user"][sp["starts"][i]])
        by_hh.setdefault(u, []).append((int(sp["day"][sp["starts"][i]]), i))
    for u in by_hh:
        by_hh[u].sort()
    elig = [u for u, v in by_hh.items() if len(v) >= a.horizon]
    rng = np.random.default_rng(a.seed)
    hh = list(rng.permutation(elig))[:a.n_households]
    hh_trips = [(int(u), [i for _, i in by_hh[u][:a.horizon]]) for u in hh]
    log(f"{len(hh_trips)} households x {a.horizon} trips = "
        f"{len(hh_trips) * a.horizon:,} generated baskets per mode")

    # the real trajectory, for reference
    real = []
    accR = {}
    for step in range(a.horizon):
        for u, t in hh_trips:
            s, e = sp["starts"][t[step]], sp["ends"][t[step]]
            acc = accR.setdefault(u, set())
            for j in sp["item"][s:e]:
                if not seen0[u, int(j)]:
                    acc.add(int(j))
        real.append(float(np.mean([len(accR[u]) for u in accR])))

    res = {"label": a.label, "n_households": len(hh_trips), "horizon": a.horizon,
           "real_cumulative_new": real, "modes": {}}
    for mode in a.modes:
        steps, secs = rollout(m, d, dev, hh_trips, mode, a.seed, a.sweeps,
                              a.item_temp, a.require_nonempty)
        cur = novelty_curve(steps, d, seen0)
        sizes = [s["mean_items"] for s in steps]
        n_b = len(hh_trips) * a.horizon
        res["modes"][mode] = {
            "cumulative_new": cur, "mean_items_by_step": sizes,
            "seconds": secs, "baskets_per_sec": n_b / max(secs, 1e-9),
            "ratio_by_step": [c / r if r > 0 else float("nan")
                              for c, r in zip(cur, real)]}
        log(f"  {mode:8s} {secs:7.1f}s  {n_b / secs:6.1f} baskets/s   "
            f"drift {cur[0] / max(real[0], 1e-9):.3f}x -> {cur[-1] / max(real[-1], 1e-9):.3f}x")

    log("")
    log(f"  {'step':>4s} {'real':>8s} " + " ".join(f"{k:>10s}" for k in res["modes"]))
    for s in range(a.horizon):
        row = f"  {s + 1:4d} {real[s]:8.2f} "
        row += " ".join(f"{res['modes'][k]['cumulative_new'][s]:10.2f}" for k in res["modes"])
        log(row)

    with open(os.path.join(OUT, f"mdp_rollout_{a.label}.json"), "w") as f:
        json.dump(res, f, indent=2)
    log("")
    log(f"wrote out/mdp_rollout_{a.label}.json")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--label", default="ps_nested")
    p.add_argument("--horizon", type=int, default=12)
    p.add_argument("--n-households", type=int, default=200)
    p.add_argument("--modes", nargs="+", default=["open", "recency", "full"])
    p.add_argument("--sweeps", type=int, default=4)
    p.add_argument("--item-temp", type=float, default=1.0,
                   help="sharpen the within-category item draw only; 0.80 calibrates the "
                        "novel-item rate without touching basket size")
    p.add_argument("--require-nonempty", action="store_true",
                   help="reject empty compositions, as spec Eq. 8 conditions on n >= 1")
    p.add_argument("--seed", type=int, default=0)
    main(p.parse_args())
