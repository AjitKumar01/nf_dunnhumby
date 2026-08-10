"""
How much of the basket-size variance has to come from a discrete trip-type mixture?

verify_response.py establishes two facts that between them force this question:

  * d E[n] / d(uniform value shift) = Var(n) exactly, so a model that under-produces
    basket-size variance under-produces the aggregate response to a chain-wide price
    move by the same factor.  Var(n)/E[n] = 10.62 on dunnhumby training baskets.

  * a single shared Gaussian factor reaches that dispersion only at lambda ~= 0.91,
    and the importance-sampling estimator is already visibly degraded by lambda ~= 0.4
    (relative error in Var 26%, ESS 0.97) and dead by lambda ~= 0.6 (two modes,
    ESS 0.25).

So the Gaussian factor must be held in the well-behaved regime and the rest of the
dispersion has to come from somewhere exactly computable.  A finite mixture over trip
types is exactly computable -- G extra terms, no approximation.  This script asks what
G has to be and what the components have to look like, by fitting a G-component mixture
directly to the observed basket-size distribution.

Writes out/size_decomposition.json.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "..", "out")


def log(m):
    print(f"[siz] {m}", flush=True)


def nb_pmf(k, mu, r):
    """Negative binomial with mean mu and dispersion r, evaluated on a grid of k."""
    from scipy.special import gammaln
    return np.exp(gammaln(k + r) - gammaln(r) - gammaln(k + 1)
                  + r * (np.log(r) - np.log(r + mu)) + k * (np.log(mu) - np.log(r + mu)))


def fit_mixture(counts, G, iters=400, seed=0):
    """EM for a G-component negative-binomial mixture on basket size, on the support
    grid with observed frequencies as weights."""
    from scipy.optimize import minimize_scalar
    k = np.arange(len(counts))
    w = counts / counts.sum()
    rng = np.random.default_rng(seed)
    mu = np.quantile(np.repeat(k, (counts * 1000).astype(int) + 1),
                     np.linspace(0.15, 0.9, G)) + rng.uniform(0, .5, G)
    r = np.full(G, 4.0)
    pi = np.full(G, 1.0 / G)
    for _ in range(iters):
        dens = np.stack([pi[g] * nb_pmf(k, mu[g], r[g]) for g in range(G)])
        tot = dens.sum(0) + 1e-300
        resp = dens / tot
        nk = (resp * w).sum(1)
        pi = nk / nk.sum()
        for g in range(G):
            ww = resp[g] * w
            mu[g] = max((ww * k).sum() / max(ww.sum(), 1e-12), 1e-3)

            def nll(logr, g=g, ww=ww):
                return -(ww * np.log(nb_pmf(k, mu[g], np.exp(logr)) + 1e-300)).sum()
            r[g] = float(np.exp(minimize_scalar(nll, bounds=(-3, 8),
                                                method="bounded").x))
    dens = np.stack([pi[g] * nb_pmf(k, mu[g], r[g]) for g in range(G)])
    fit = dens.sum(0)
    return pi, mu, r, fit


def main(a):
    b = pd.read_parquet(os.path.join(HERE, "..", "..", "basket_input", "baskets.parquet"))
    tr = b[b.split == "train"]
    n = tr.groupby("BASKET_ID").size().values
    E, V = float(n.mean()), float(n.var())
    log(f"observed basket size: E[n] {E:.4f}  Var(n) {V:.4f}  dispersion {V / E:.4f}")

    kmax = int(n.max())
    counts = np.bincount(n, minlength=kmax + 1).astype(float)
    emp = counts / counts.sum()

    res = {"observed": {"mean": E, "var": V, "dispersion": V / E, "max": kmax}}
    log("")
    log(f"  {'G':>2} {'TVD':>7} {'within Var':>11} {'between Var':>12} "
        f"{'between share':>14}")
    rows = []
    for G in range(1, a.gmax + 1):
        pi, mu, r, fit = fit_mixture(counts, G, seed=a.seed)
        tvd = 0.5 * np.abs(fit - emp).sum()
        within = float((pi * (mu + mu ** 2 / r)).sum())
        between = float((pi * (mu - (pi * mu).sum()) ** 2).sum())
        rows.append(dict(G=G, tvd=tvd, within=within, between=between,
                         share=between / (within + between),
                         weights=[round(float(x), 4) for x in pi],
                         means=[round(float(x), 3) for x in mu],
                         disp_r=[round(float(x), 3) for x in r]))
        log(f"  {G:2d} {tvd:7.4f} {within:11.3f} {between:12.3f} "
            f"{between / (within + between):13.1%}")
    res["mixtures"] = rows

    log("")
    best = min(rows, key=lambda d: d["tvd"])
    log(f"best fit at G={best['G']}, total variation {best['tvd']:.4f}")
    log(f"  weights {best['weights']}")
    log(f"  means   {best['means']}")
    log("")
    # what the Gaussian factor would have to carry if the mixture carries the rest
    for G in (1, 2, 3):
        row = next(x for x in rows if x["G"] == G)
        log(f"  G={G}: the shared Gaussian factor must supply within-component "
            f"dispersion {row['within'] / E:.2f}")
    log("")
    log("A within-component dispersion near 1 is the regime where the mode map is a")
    log("strong contraction and importance sampling is exact to three decimals.")
    with open(os.path.join(OUT, "size_decomposition.json"), "w") as fh:
        json.dump(res, fh, indent=2)
    log("wrote out/size_decomposition.json")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--gmax", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    main(p.parse_args())
