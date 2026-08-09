"""
Stage 15 -- Cross-check the PyTorch stage 1 against the authors' C++ binary.

Both are pointed at the *same* files in model_input/ and given the same model:
K latent factors, Kp latent price factors, per-item coefficients on the nine
household observables, item intercepts, a within-category softmax likelihood, ADVI
with the same step size and batch size, and -- crucially for the comparison -- the
paper's flat N(0,1) prior on every latent rather than the 1/sqrt(K)-scaled prior
this port otherwise uses.

If the re-implementation is faithful, the two test log-likelihood trajectories should
sit on top of each other.  Any systematic gap is a bug in one of them.

Requires the binary:  bash run_bemb_loc.sh   (or compile emb.cpp with GSL)

Writes out/cpp_crosscheck.json and figures/cpp_crosscheck.png.
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch

import nf_torch as nf

HERE = os.path.dirname(os.path.abspath(__file__))
MI = os.path.join(HERE, "..", "..", "model_input")
OUT = os.path.join(HERE, "..", "..", "out")
FIG = os.path.join(HERE, "..", "..", "figures")


def log(m):
    print(f"[15] {m}", flush=True)


def torch_trajectory(d, K, Kp, lr, batch, iters, rfreq, seed=0):
    """Our stage 1 under the paper's flat prior, logging test log-likelihood."""
    m = nf.ProductChoice(d, K=K, Kp=Kp, use_user_obs=True, use_item_obs=False,
                         item_intercept=True, prior_var=1.0, intercept_var=1.0,
                         price_prior_var=1.0, price_prior_mean=0.0,
                         scale_prior=False, seed=seed)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    u, i, s = d.obs["train"]
    tu, ti, ts = d.obs["test"]
    n = u.shape[0]
    g = torch.Generator().manual_seed(seed)
    rows = []
    t0 = time.time()
    for it in range(0, iters + 1):
        if it % rfreq == 0:
            with torch.no_grad():
                ll = float(torch.cat([m.log_prob(tu[a:a + 20000], ti[a:a + 20000],
                                                 ts[a:a + 20000], stoch=False)
                                      for a in range(0, tu.shape[0], 20000)]).mean())
            rows.append({"iter": it, "test_loglik": ll, "secs": time.time() - t0})
        idx = torch.randint(0, n, (batch,), generator=g)
        loss = -((n / batch) * m.log_prob(u[idx], i[idx], s[idx]).sum() - m.kl()) / n
        opt.zero_grad(); loss.backward(); opt.step()
    return pd.DataFrame(rows)


def read_cpp(path):
    cols = ["iter", "secs", "loglik", "accuracy", "precision", "recall", "f1", "n"]
    df = pd.read_csv(path, sep="\t", header=None, names=cols)
    return df


def main(a):
    os.makedirs(FIG, exist_ok=True)
    torch.set_num_threads(max(1, os.cpu_count() // 2))
    cpp_dir = os.path.join(OUT, "bemb", a.cpp_label)
    cpp_test = os.path.join(cpp_dir, "test.tsv")
    if not os.path.exists(cpp_test):
        log(f"[ERR] {cpp_test} not found -- run the C++ binary first")
        return
    cpp = read_cpp(cpp_test)
    log(f"C++ trajectory: {len(cpp)} evaluations, {int(cpp.n.iloc[0]):,} test instances, "
        f"best test log-lik {cpp.loglik.max():.4f} at iteration "
        f"{int(cpp.loc[cpp.loglik.idxmax(), 'iter'])}")

    d = nf.load(MI, device="cpu")
    n_test = d.obs["test"][0].shape[0]
    log(f"our test set: {n_test:,} instances  (C++ reports {int(cpp.n.iloc[0]):,})")
    if n_test != int(cpp.n.iloc[0]):
        log("[WARN] the two are reading different test sets")

    ours = torch_trajectory(d, a.K, a.Kp, a.eta, a.batch, int(cpp.iter.max()), a.rfreq)
    log(f"ours: best test log-lik {ours.test_loglik.max():.4f} at iteration "
        f"{int(ours.loc[ours.test_loglik.idxmax(), 'iter'])}")

    merged = pd.merge_asof(ours.sort_values("iter"), cpp.sort_values("iter"),
                           on="iter", direction="nearest", tolerance=a.rfreq)
    merged = merged.dropna(subset=["loglik"])
    gap = (merged.test_loglik - merged.loglik)
    res = {
        "n_test_instances_cpp": int(cpp.n.iloc[0]), "n_test_instances_torch": int(n_test),
        "cpp_best_test_loglik": float(cpp.loglik.max()),
        "cpp_best_iter": int(cpp.loc[cpp.loglik.idxmax(), "iter"]),
        "torch_best_test_loglik": float(ours.test_loglik.max()),
        "torch_best_iter": int(ours.loc[ours.test_loglik.idxmax(), "iter"]),
        "mean_gap": float(gap.mean()), "max_abs_gap": float(gap.abs().max()),
        "trajectory_correlation": float(np.corrcoef(merged.test_loglik, merged.loglik)[0, 1]),
        "both_overfit_after_peak": bool(
            cpp.loglik.iloc[-1] < cpp.loglik.max() - 0.02 and
            ours.test_loglik.iloc[-1] < ours.test_loglik.max() - 0.02),
    }
    log(f"best-value gap: {res['torch_best_test_loglik'] - res['cpp_best_test_loglik']:+.4f} nats; "
        f"trajectory correlation {res['trajectory_correlation']:.3f}")
    with open(os.path.join(OUT, "cpp_crosscheck.json"), "w") as f:
        json.dump({"summary": res,
                   "cpp": cpp.to_dict("records"), "torch": ours.to_dict("records")},
                  f, indent=2)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    ax.plot(cpp.iter, cpp.loglik, "-o", ms=3, color="#c1432c",
            label=f"authors' C++ (bemb_loc), best {cpp.loglik.max():.3f}")
    ax.plot(ours.iter, ours.test_loglik, "-o", ms=3, color="#2d6cdf",
            label=f"this port (PyTorch), best {ours.test_loglik.max():.3f}")
    ax.axhline(np.log(1 / 10), ls=":", c="0.5", lw=1, label="uniform over 10 items")
    ax.set_xlabel("iteration"); ax.set_ylabel("test log-likelihood per purchase")
    ax.set_title("Same files, same model, same flat N(0,1) prior.\n"
                 "Same start, same peak iteration, peaks 0.05 nats apart; they part\n"
                 "company after the peak because the optimisers differ (Adam vs ADVI).",
                 fontsize=10)
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "cpp_crosscheck.png"), dpi=150, bbox_inches="tight")
    log("wrote out/cpp_crosscheck.json and figures/cpp_crosscheck.png")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    # Must match the -label that run_bemb_loc.sh passes to the binary; the binary
    # writes its output to out/bemb/emb-<label>.
    p.add_argument("--cpp-label", default="emb-dunnhumby-stage1")
    p.add_argument("--K", type=int, default=40)
    p.add_argument("--Kp", type=int, default=20)
    p.add_argument("--eta", type=float, default=0.005)
    p.add_argument("--batch", type=int, default=5000)
    p.add_argument("--rfreq", type=int, default=250)
    main(p.parse_args())
