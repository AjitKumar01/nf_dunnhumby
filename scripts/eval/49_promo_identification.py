"""
Stage 49 -- What happens to the price coefficient when promotion is controlled for.

The placebo in Section 6 of the experiments page rules out a spurious constant: scramble
the price panel and the coefficient goes to zero.  What it cannot do is separate "price
moves demand" from "the retailer times promotions to demand", because permuting a price
series destroys both.  That is the open identification problem, and it is open because
nothing in basket_input carried a promotion signal.

dunnhumby's causal_data.csv does.  Stage 23 turns it into a binary (item, store, week)
panel of display and mailer placement, and stage 27 --use-promo conditions on it.  The
comparison here is the direct test:

    if the fitted price coefficient falls when placement is controlled for, then part of
    what was being attributed to PRICE was really PROMOTION -- the confound the placebo
    cannot see.

This is a control, not an instrument, and the distinction matters.  Being in the weekly
circular plausibly raises demand on its own -- that is what advertising is -- so mailer
fails the exclusion restriction and cannot instrument for price.  What it can do is
absorb the promotion channel so that what remains on the price coefficient is closer to
a pure price response.  The residual is still not identified against unobserved demand
shocks the retailer responds to.

Writes out/promo_identification.json.
"""
import argparse
import importlib
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "model"))
nb = importlib.import_module("27_nested_basket")
cf = importlib.import_module("28_nested_counterfactual")

HERE = os.path.dirname(os.path.abspath(__file__))
IN = os.path.join(HERE, "..", "..", "basket_input")
OUT = os.path.join(HERE, "..", "..", "out")


def log(m):
    print(f"[49] {m}", flush=True)


def main(a):
    dev = torch.device("cpu")
    d = nb.NestedData(IN, device=dev)
    res = {"control": a.control, "treated": a.treated, "models": {}}

    for lab in (a.control, a.treated):
        m, _ = cf.load(lab, d, dev)
        h = json.load(open(os.path.join(OUT, f"{lab}_nested_history.json")))
        with torch.no_grad():
            g = (m.gamma @ m.beta.T)
            gq = (m.q_gamma @ m.q_beta.T) if m.use_quantity else torch.zeros(1)
            row = {"test_item": h["test_item"], "test_top1": h["test_top1"],
                   "test_incidence_nll": h["test_incidence_nll"],
                   "price_median": float(g.median()), "price_mean": float(g.mean()),
                   "qprice_median": float(gq.median()),
                   "kappa_median": float(m.kappa().median()),
                   "frac_wrong_sign": float((g < 0).float().mean())}
            if getattr(m, "w_promo", None) is not None:
                w = m.w_promo.detach()
                row["w_display_mean"] = float(w[:, 0].mean())
                row["w_mailer_mean"] = float(w[:, 1].mean())
                row["w_display_sd"] = float(w[:, 0].std())
                row["w_mailer_sd"] = float(w[:, 1].std())
        res["models"][lab] = row

    c, t = res["models"][a.control], res["models"][a.treated]
    drop = 1.0 - t["price_median"] / c["price_median"]
    res["price_coefficient_drop"] = float(drop)
    res["item_gain"] = float(t["test_item"] - c["test_item"])

    log("")
    log(f"  {'':28s} {'no control':>12s} {'+ promotion':>12s}")
    for k, nm in (("test_item", "item log-lik"), ("test_top1", "top-1"),
                  ("test_incidence_nll", "incidence NLL"),
                  ("price_median", "price coef (median g)"),
                  ("qprice_median", "quantity price coef"),
                  ("kappa_median", "kappa"), ("frac_wrong_sign", "wrong-sign share")):
        log(f"  {nm:28s} {c[k]:12.4f} {t[k]:12.4f}")
    log("")
    log(f"  price coefficient falls {c['price_median']:.4f} -> {t['price_median']:.4f} "
        f"({100 * drop:+.1f}%)")
    log(f"  item log-likelihood moves {res['item_gain']:+.4f} "
        f"(refit noise is 0.0110)")
    if "w_display_mean" in t:
        log("")
        log(f"  fitted placement loadings, per product:")
        log(f"    display  mean {t['w_display_mean']:+.4f}  sd {t['w_display_sd']:.4f}")
        log(f"    mailer   mean {t['w_mailer_mean']:+.4f}  sd {t['w_mailer_sd']:.4f}")

    log("")
    if drop > 0.10:
        log(f"  READING: {100 * drop:.0f}% of the price coefficient was promotion.")
        log("  The placebo could not have found this: scrambling prices destroys the")
        log("  promotion association too, so it collapses either way.")
    else:
        log("  READING: controlling for placement leaves the price coefficient intact,")
        log("  so promotion timing was not driving it.")
    log("")
    log("  NOT an instrument.  Being in the circular is advertising, which moves demand")
    log("  on its own, so mailer fails exclusion.  This absorbs the promotion channel;")
    log("  it does not identify price against unobserved demand shocks.")

    with open(os.path.join(OUT, "promo_identification.json"), "w") as f:
        json.dump(res, f, indent=2)
    log("")
    log("wrote out/promo_identification.json")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--control", default="ps_nested")
    p.add_argument("--treated", default="pr_on")
    main(p.parse_args())
