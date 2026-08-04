"""
Stage 31 -- Benchmarks: is the nested model better than something simpler?

Auditing what the nested model had been compared against turned up an awkward answer:
its own ablations, its own structural placebo, and an embedding-only comparison with
the paper's model.  All of those say "each component earns its place" or "the price
coefficient is real".  None of them says the model is better than a simpler thing at
the task it exists to do.

An ablation is not a benchmark.  Removing a head from a 900k-parameter model and
watching the score drop shows the head is used; it does not show the model beats a
baseline anyone would actually consider.

So this scores everything through **one scorer, on one candidate set**.  That matters
more than it sounds: the nested model masks unstocked items out of the choice set,
which makes its softmax denominator smaller and its log-likelihood mechanically
higher.  Comparing its reported number against a model fitted without that mask would
credit it for an easier question (NESTED_MODEL.md 8.2 measures that at 99.7% of the
apparent store gain).  Here every model is scored on **identical candidate sets** --
same positives, same negatives, same availability rule -- so the numbers are
comparable by construction.

Baselines, cheapest first:

  random          uniform over the candidate set.  The floor.
  popularity      global purchase counts.  Note that negatives are drawn
                  unigram^0.75, i.e. popularity-weighted, so this baseline is being
                  asked to separate a popular true item from popular decoys -- a
                  deliberately hard test rather than a free win.  Its temperature is
                  tuned on validation, because an untuned log-count score is peaked
                  enough that being confidently wrong scores *below* uniform.
  household       what this household bought before, backed off to popularity.
                  This is the baseline a recommender practitioner would reach for,
                  and it is strong because grocery is repetitive.
  co-occurrence   item-item counts from the training baskets, scored against what is
                  already in the basket.  A model-free version of the interaction
                  term, so it tests whether the learned embedding beats counting.
  flat basket     the model in BASKET_MODEL.md, re-scored on this candidate set.
  nested          the model in NESTED_MODEL.md.

Writes out/benchmark.json and figures/benchmark.png.
"""
import argparse
import importlib
import json
import os

import numpy as np
import pandas as pd
import torch

nb = importlib.import_module("27_nested_basket")

HERE = os.path.dirname(os.path.abspath(__file__))
IN = os.path.join(HERE, "..", "basket_input")
OUT = os.path.join(HERE, "..", "out")
FIG = os.path.join(HERE, "..", "figures")

PAL = {"blue": "#2d6cdf", "grey": "#9aa5b1", "red": "#d1495b",
       "green": "#2a9d8f", "amber": "#e9c46a", "purple": "#7b6cd9"}


def log(m):
    print(f"[31] {m}", flush=True)


def build_batches(d, split, n_batches, n_neg, seed, device, ctx_model,
                  batch_baskets=256):
    """Fixed candidate sets, generated once and reused by every model.

    This is the whole point of the script.  Each model must see the same positives,
    the same negatives and the same availability mask, or the comparison measures the
    evaluation protocol rather than the models.
    """
    rng = np.random.default_rng(seed)
    sp = d.splits[split]
    # The batch builder computes the basket context from a model's own alpha, so a
    # stub with zero alpha silently hands every model a zero context -- which
    # disables the nested model's interaction term at scoring time and makes it look
    # worse than its own no-interaction ablation.  Each fitted model therefore gets
    # batches built with *its* alpha; the heuristic baselines do not use context at
    # all and are unaffected by which set they are scored on.
    out = []
    for _ in range(n_batches):
        bidx = rng.integers(0, sp["n_baskets"], size=batch_baskets)
        bt = nb.make_batch(d, ctx_model, split, bidx, n_neg, rng, device)
        out.append((bidx, bt))
    return out


def score_loglik(scores, avail, rng=None):
    """Mean log P(true item), where the true item is column 0.

    Ties are broken at random, not by position.  The true item sits in column 0, so
    `argmax` on an all-equal row returns 0 and a model with no information scores
    top-1 = 1.000 -- which is exactly what the random baseline did on the first run.
    A tiny random jitter, far below any real score difference, removes the artefact.
    """
    s = scores.masked_fill(~avail, -1e9)
    lp = torch.log_softmax(s, dim=1)[:, 0]
    j = torch.rand(s.shape, device=s.device, generator=rng) * 1e-6
    return float(lp.mean()), float(((s + j).argmax(1) == 0).float().mean())


def main(a):
    os.makedirs(FIG, exist_ok=True)
    dev = torch.device(a.device)
    d = nb.NestedData(IN, device=dev)
    bk = pd.read_parquet(os.path.join(IN, "baskets.parquet"))
    tr = bk[bk.split == "train"]

    # Tune the heuristic weights on VALIDATION before scoring on test.  Guessing them
    # would understate the baselines and flatter the fitted model -- the fitted model
    # had its own hyperparameters selected, so the baselines must too, or the
    # comparison is not like for like.
    if not a.no_tune:
        stub0 = type("S", (), {"K": 1, "alpha": torch.zeros(d.J, 1, device=dev),
                               "use_store": True, "use_store_price": True})()
        vb = build_batches(d, "validation", 6, a.n_neg, a.seed + 1, dev, stub0)
        cnt0 = np.bincount(tr.item_id.to_numpy(), minlength=d.J).astype(np.float64)
        pop0 = torch.as_tensor(np.log(cnt0 + 1.0), dtype=torch.float32, device=dev)
        hh0 = tr.groupby(["user_id", "item_id"]).size()
        H0 = torch.zeros(d.N, d.J, device=dev)
        H0[torch.as_tensor(np.array([i for i, _ in hh0.index]), device=dev),
           torch.as_tensor(np.array([j for _, j in hh0.index]), device=dev)] = \
            torch.as_tensor(np.log1p(hh0.to_numpy()), dtype=torch.float32, device=dev)
        # popularity temperature first
        bp, bwp = -1e9, 1.0
        for w in [0.05, 0.1, 0.25, 0.5, 1.0]:
            v = np.mean([score_loglik(pop0[bt["cand"]] * w, bt["avail"])[0]
                         for _, bt in vb])
            if v > bp:
                bp, bwp = v, w
        a.w_pop = bwp
        log(f"tuned on validation: w_pop = {bwp} (val log-lik {bp:.4f})")
        best, bw = -1e9, a.w_repeat
        for w in [0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0]:
            v = np.mean([score_loglik(H0[bt["user"].unsqueeze(1), bt["cand"]] * w
                                      + pop0[bt["cand"]] * a.w_pop, bt["avail"])[0]
                         for _, bt in vb])
            if v > best:
                best, bw = v, w
        a.w_repeat = bw
        log(f"tuned on validation: w_repeat = {bw} (val log-lik {best:.4f})")

    stub = type("S", (), {"K": 1, "alpha": torch.zeros(d.J, 1, device=dev),
                          "use_store": True, "use_store_price": True})()
    batches = build_batches(d, "test", a.batches, a.n_neg, a.seed, dev, stub)
    n_rows = sum(b[1]["cand"].shape[0] for b in batches)
    log(f"scoring {len(batches)} batches, {n_rows:,} held-out purchases, "
        f"{a.n_neg} negatives each -- identical candidate sets for every model")

    res = {}

    # ---------------------------------------------------------------- random
    tot = [score_loglik(torch.zeros_like(bt["dlogp"]), bt["avail"]) for _, bt in batches]
    res["random"] = {"loglik": float(np.mean([x[0] for x in tot])),
                     "top1": float(np.mean([x[1] for x in tot]))}

    # ------------------------------------------------------------ popularity
    cnt = np.bincount(tr.item_id.to_numpy(), minlength=d.J).astype(np.float64)
    pop = torch.as_tensor(np.log(cnt + 1.0), dtype=torch.float32, device=dev)
    tot = [score_loglik(pop[bt["cand"]] * a.w_pop, bt["avail"]) for _, bt in batches]
    res["popularity"] = {"loglik": float(np.mean([x[0] for x in tot])),
                         "top1": float(np.mean([x[1] for x in tot]))}

    # -------------------------------------------------- household repeat-purchase
    # log(1 + times this household bought this item in training), plus a small
    # popularity term so unseen items are ranked sensibly rather than tied at zero.
    hh = tr.groupby(["user_id", "item_id"]).size()
    H = torch.zeros(d.N, d.J, device=dev)
    ui = np.array([i for i, _ in hh.index]); ii = np.array([j for _, j in hh.index])
    H[torch.as_tensor(ui, device=dev), torch.as_tensor(ii, device=dev)] = \
        torch.as_tensor(np.log1p(hh.to_numpy()), dtype=torch.float32, device=dev)
    tot = []
    for _, bt in batches:
        s = H[bt["user"].unsqueeze(1), bt["cand"]] * a.w_repeat + pop[bt["cand"]] * a.w_pop
        tot.append(score_loglik(s, bt["avail"]))
    res["household_repeat"] = {"loglik": float(np.mean([x[0] for x in tot])),
                               "top1": float(np.mean([x[1] for x in tot]))}

    # ------------------------------------------------------- item-item co-occurrence
    # A model-free stand-in for the interaction term: how often does this candidate
    # appear in a basket alongside the items already in this basket?
    log("building item-item co-occurrence from training baskets ...")
    g = tr.groupby("BASKET_ID").item_id.apply(list)
    from collections import Counter
    co = Counter()
    for its in g:
        u = sorted(set(its))
        if len(u) > 60:
            continue
        for x in range(len(u)):
            for y in range(x + 1, len(u)):
                co[(u[x], u[y])] += 1
                co[(u[y], u[x])] += 1
    idx = np.array(list(co.keys()), dtype=np.int64)
    val = np.array(list(co.values()), dtype=np.float32)
    C = torch.sparse_coo_tensor(torch.as_tensor(idx.T, device=dev),
                                torch.as_tensor(np.log1p(val), device=dev),
                                (d.J, d.J)).coalesce().to_dense()
    log(f"  {len(co):,} nonzero item pairs")
    sp_t = d.splits["test"]
    tot = []
    for bidx, bt in batches:
        # mean co-occurrence with the other items of the same basket
        starts, ends = sp_t["starts"][bidx], sp_t["ends"][bidx]
        rows = np.concatenate([np.arange(s, e) for s, e in zip(starts, ends)])
        owner = np.repeat(np.arange(len(bidx)), ends - starts)
        basket_vec = torch.zeros(len(bidx), d.J, device=dev)
        basket_vec[torch.as_tensor(owner, device=dev),
                   torch.as_tensor(sp_t["item"][rows], device=dev)] = 1.0
        # Mean, not sum: a basket of 30 items would otherwise score 30x a basket of
        # one, and log1p counts summed over a basket reach 50-100, which saturates the
        # softmax so completely that a single wrong prediction dominates the average.
        n_in = basket_vec.sum(1, keepdim=True).clamp_min(1)
        ctx = (basket_vec @ C) / n_in                           # [B, J]
        s = torch.gather(ctx[torch.as_tensor(owner, device=dev)], 1, bt["cand"])
        s = s * a.w_cooc + pop[bt["cand"]] * a.w_pop
        tot.append(score_loglik(s, bt["avail"]))
    res["cooccurrence"] = {"loglik": float(np.mean([x[0] for x in tot])),
                           "top1": float(np.mean([x[1] for x in tot]))}

    # ------------------------------------------------- household + co-occurrence
    tot = []
    for bidx, bt in batches:
        starts, ends = sp_t["starts"][bidx], sp_t["ends"][bidx]
        rows = np.concatenate([np.arange(s, e) for s, e in zip(starts, ends)])
        owner = np.repeat(np.arange(len(bidx)), ends - starts)
        basket_vec = torch.zeros(len(bidx), d.J, device=dev)
        basket_vec[torch.as_tensor(owner, device=dev),
                   torch.as_tensor(sp_t["item"][rows], device=dev)] = 1.0
        n_in = basket_vec.sum(1, keepdim=True).clamp_min(1)
        ctx = (basket_vec @ C) / n_in
        s = (H[bt["user"].unsqueeze(1), bt["cand"]] * a.w_repeat
             + torch.gather(ctx[torch.as_tensor(owner, device=dev)], 1, bt["cand"]) * a.w_cooc
             + pop[bt["cand"]] * a.w_pop)
        tot.append(score_loglik(s, bt["avail"]))
    res["household_plus_cooc"] = {"loglik": float(np.mean([x[0] for x in tot])),
                                  "top1": float(np.mean([x[1] for x in tot]))}

    # ------------------------------------------------------------- fitted models
    for label, kind in [("nested", "nested"), ("nested_noctx", "nested")]:
        path = os.path.join(OUT, f"{label}_nested.pt")
        if not os.path.exists(path):
            continue
        cfg = json.load(open(os.path.join(OUT, f"{label}_nested_history.json")))["config"]
        m = nb.NestedModel(d, K=cfg["K"], Kp=cfg["Kp"], Kt=cfg["Kt"], Ks=cfg["Ks"],
                           seed=cfg["seed"], use_nest=not cfg["no_nest"],
                           use_quantity=not cfg["no_quantity"],
                           use_store=not cfg["no_store"],
                           use_breadth=not cfg.get("no_breadth", False),
                           use_context=not cfg.get("no_context", False)).to(dev)
        m.load_state_dict(torch.load(path, map_location=dev)); m.eval()
        # rebuild with this model's own alpha, so its interaction term is live
        mb = build_batches(d, "test", a.batches, a.n_neg, a.seed, dev, m)
        with torch.no_grad():
            tot = []
            for _, bt in mb:
                s = m.item_utility(bt["user"], bt["cand"], bt["ctx"], bt["dlogp"],
                                   bt["state"], bt["week"], bt["store"])
                tot.append(score_loglik(s, bt["avail"]))
        res[label] = {"loglik": float(np.mean([x[0] for x in tot])),
                      "top1": float(np.mean([x[1] for x in tot]))}

    # ------------------------------------------------------------------ report
    order = sorted(res, key=lambda k: res[k]["loglik"])
    log("")
    log(f"{'model':22s} {'log-lik':>9s} {'top-1':>7s}   vs popularity")
    base = res["popularity"]["loglik"]
    for k in order:
        v = res[k]
        log(f"{k:22s} {v['loglik']:9.4f} {v['top1']:7.3f}   {v['loglik'] - base:+.4f}")
    res["_meta"] = {"held_out_purchases": int(n_rows), "n_neg": a.n_neg,
                    "chance_loglik": float(np.log(1.0 / (a.n_neg + 1)))}
    with open(os.path.join(OUT, "benchmark.json"), "w") as f:
        json.dump(res, f, indent=2)

    # ------------------------------------------------------------------ figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    names = [k for k in order]
    cols = [PAL["blue"] if k.startswith("nested") else PAL["grey"] for k in names]
    ax = axes[0]
    ax.barh(range(len(names)), [res[k]["loglik"] for k in names], color=cols)
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=8)
    ax.axvline(res["_meta"]["chance_loglik"], color=PAL["red"], ls="--", lw=1,
               label="uniform over the candidate set")
    ax.set_xlabel("held-out log-likelihood (higher is better)")
    ax.set_title("Every model on identical candidate sets", fontsize=10)
    ax.legend(fontsize=8); ax.grid(axis="x", alpha=.3)
    for i, k in enumerate(names):
        ax.text(res[k]["loglik"], i, f"  {res[k]['loglik']:.3f}", va="center", fontsize=8)
    ax = axes[1]
    ax.barh(range(len(names)), [res[k]["top1"] for k in names], color=cols)
    ax.set_yticks(range(len(names))); ax.set_yticklabels([], fontsize=8)
    ax.axvline(1.0 / (a.n_neg + 1), color=PAL["red"], ls="--", lw=1, label="chance")
    ax.set_xlabel(f"top-1 accuracy among 1 + {a.n_neg} candidates")
    ax.set_title("Ranking the true item first", fontsize=10)
    ax.legend(fontsize=8); ax.grid(axis="x", alpha=.3)
    for i, k in enumerate(names):
        ax.text(res[k]["top1"], i, f"  {res[k]['top1']:.3f}", va="center", fontsize=8)
    fig.suptitle("Benchmarks: is the model better than something simpler?", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "benchmark.png"), dpi=150, bbox_inches="tight")
    log("")
    log("wrote out/benchmark.json and figures/benchmark.png")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--batches", type=int, default=24)
    p.add_argument("--n-neg", type=int, default=20)
    p.add_argument("--w-repeat", type=float, default=1.0,
                   help="scale on the repeat-purchase score; tuned on validation "
                        "unless --no-tune, because a guessed scale would understate "
                        "the baseline and flatter the fitted model")
    p.add_argument("--w-cooc", type=float, default=1.0)
    p.add_argument("--w-pop", type=float, default=0.25)
    p.add_argument("--no-tune", action="store_true")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--device", default="cpu")
    main(p.parse_args())
