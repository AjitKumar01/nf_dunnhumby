"""
Stage 10 -- Audit of the price reconstruction against the dunnhumby user guide.

transaction_data carries three separate discount columns and no price column, so the
price the model sees is a construction and has to be defended.  This script

  * checks the sign convention of all three discount columns,
  * reproduces the three worked examples on p.3 of the user guide from real rows,
  * shows that the guide's two named formulas are inconsistent with its own worked
    examples, and which of the two the examples support,
  * compares four candidate price definitions on how well each behaves like a
    *posted shelf price* -- concentration at a modal value among shoppers facing the
    same shelf on the same day,
  * measures how much each discount actually moves the number.

Writes out/price_audit.json and figures/price_definitions.png.
"""
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
OUT = os.path.join(HERE, "..", "out")
FIG = os.path.join(HERE, "..", "figures")
# Raw dunnhumby CSVs.  Defaults to a sibling of the repository; override with
# NF_RAW_DIR if the download lives somewhere else.
RAW = os.path.join(os.environ.get(
    "NF_RAW_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                 "dunnhumby_The-Complete-Journey",
                 "dunnhumby_The-Complete-Journey CSV")), "")

# The four candidates.  SALES_VALUE is stated by the guide to be the money the
# retailer receives, already net of the loyalty discount and of the retailer's match
# of a manufacturer coupon, and inclusive of the manufacturer's reimbursement.
# Discounts are stored as negative numbers, so subtracting one adds it back.
DEFS = {
    "A_sales_value": lambda d: d.SALES_VALUE / d.QUANTITY,
    "B_loyalty_shelf": lambda d: (d.SALES_VALUE - d.COUPON_MATCH_DISC) / d.QUANTITY,
    "C_regular_shelf": lambda d: (d.SALES_VALUE - d.RETAIL_DISC - d.COUPON_MATCH_DISC) / d.QUANTITY,
    "D_customer_paid": lambda d: (d.SALES_VALUE + d.COUPON_DISC) / d.QUANTITY,
}
LABELS = {
    "A_sales_value": "A  SALES_VALUE / Q\n(retailer receipts)",
    "B_loyalty_shelf": "B  (SALES_VALUE - COUPON_MATCH) / Q\n(posted loyalty price)",
    "C_regular_shelf": "C  (SALES_VALUE - RETAIL_DISC\n     - COUPON_MATCH) / Q  (regular shelf)",
    "D_customer_paid": "D  (SALES_VALUE + COUPON_DISC) / Q\n(what the shopper paid)",
}


def log(m):
    print(f"[10] {m}", flush=True)


def modal_share(v, ndigits=2):
    c = np.round(np.asarray(v, dtype=float), ndigits)
    if len(c) == 0:
        return np.nan
    _, counts = np.unique(c, return_counts=True)
    return counts.max() / len(c)


def main():
    os.makedirs(FIG, exist_ok=True)
    os.makedirs(OUT, exist_ok=True)
    tx = pd.read_csv(RAW + "transaction_data.csv")
    r = {}

    # ------------------------------------------------------------ sign convention
    r["sign_convention"] = {
        c: {"min": float(tx[c].min()), "max": float(tx[c].max()),
            "share_negative": float((tx[c] < 0).mean()),
            "share_positive": float((tx[c] > 0).mean()),
            "share_zero": float((tx[c] == 0).mean())}
        for c in ["RETAIL_DISC", "COUPON_DISC", "COUPON_MATCH_DISC"]
    }
    for c, v in r["sign_convention"].items():
        log(f"{c:>18}: nonzero on {1 - v['share_zero']:.4f} of lines, "
            f"range [{v['min']:.2f}, {v['max']:.2f}], positive on {v['share_positive']:.5f}")

    # ------------------------------------------- reproduce the guide's examples
    # p.3, line 2: two units, sales_value $2.00, retail_disc $1.34, no coupon.
    # p.3, line 3: two units, sales_value $2.89, retail_disc $0.45, coupon_disc $0.55.
    ex = {}
    e2 = tx[(tx.QUANTITY == 2) & (np.isclose(tx.SALES_VALUE, 2.00)) &
            (np.isclose(tx.RETAIL_DISC, -1.34)) & (tx.COUPON_DISC == 0)]
    e3 = tx[(tx.QUANTITY == 2) & (np.isclose(tx.SALES_VALUE, 2.89)) &
            (np.isclose(tx.RETAIL_DISC, -0.45))]
    if len(e3) and (e3.COUPON_DISC < 0).any():
        e3 = e3[e3.COUPON_DISC < 0]
    for name, e, target in [("guide_line2", e2, {"regular_shelf": 1.67, "loyalty_shelf": 1.00}),
                            ("guide_line3", e3, {"regular_shelf": 1.67, "customer_paid_total": 2.34})]:
        if len(e) == 0:
            ex[name] = {"found": 0}
            continue
        row = e.iloc[0]
        ex[name] = {
            "found": int(len(e)),
            "SALES_VALUE": float(row.SALES_VALUE), "QUANTITY": int(row.QUANTITY),
            "RETAIL_DISC": float(row.RETAIL_DISC), "COUPON_DISC": float(row.COUPON_DISC),
            "COUPON_MATCH_DISC": float(row.COUPON_MATCH_DISC),
            "regular_shelf_computed": float((row.SALES_VALUE - row.RETAIL_DISC
                                             - row.COUPON_MATCH_DISC) / row.QUANTITY),
            "loyalty_shelf_computed": float((row.SALES_VALUE - row.COUPON_MATCH_DISC)
                                            / row.QUANTITY),
            "customer_paid_total_computed": float(row.SALES_VALUE + row.COUPON_DISC),
            "guide_says": target,
        }
        log(f"{name}: regular shelf computed "
            f"{ex[name]['regular_shelf_computed']:.2f} (guide {target.get('regular_shelf')}), "
            f"loyalty shelf computed {ex[name]['loyalty_shelf_computed']:.2f}"
            + (f" (guide {target['loyalty_shelf']})" if 'loyalty_shelf' in target else "")
            + (f", customer paid {ex[name]['customer_paid_total_computed']:.2f} "
               f"(guide {target['customer_paid_total']})" if 'customer_paid_total' in target else ""))
    r["guide_examples"] = ex

    # The guide labels (sales_value - (retail_disc + coupon_match_disc))/quantity as
    # the *loyalty* price, but on its own line-2 example that expression returns
    # $1.67, which the same paragraph calls the price "exclusive of loyalty card
    # discount".  With discounts stored negative the two labels are swapped; the
    # worked examples are the authority.
    r["guide_formula_labels_consistent_with_examples"] = False

    # --------------------------------------------------- candidate comparison
    tx = tx[(tx.QUANTITY > 0) & (tx.SALES_VALUE > 0) & (tx.QUANTITY <= 30)].copy()
    for k, f in DEFS.items():
        tx[k] = f(tx)

    # A posted shelf price is the same for every card holder buying that product that
    # day.  Restrict to single-unit purchases so multi-buy mechanics cannot explain
    # dispersion, and to product-days with enough buyers to measure concentration.
    one = tx[tx.QUANTITY == 1]
    grp = one.groupby(["PRODUCT_ID", "DAY"])
    size = grp.size()
    dense = size[size >= 8].index
    sub = one.set_index(["PRODUCT_ID", "DAY"]).loc[dense].reset_index()
    log(f"concentration measured on {len(dense):,} product-days with >=8 single-unit buyers "
        f"({len(sub):,} lines)")

    conc = {}
    for k in DEFS:
        conc[k] = float(sub.groupby(["PRODUCT_ID", "DAY"])[k].apply(modal_share).mean())
        log(f"  {k:>16}: mean modal-price share = {conc[k]:.4f}")
    r["modal_share_by_definition"] = conc
    r["n_product_days"] = int(len(dense))

    # How much does each discount move the number, when it is active?
    act = {
        "retail_disc_active_share": float((tx.RETAIL_DISC < 0).mean()),
        "coupon_match_active_share": float((tx.COUPON_MATCH_DISC < 0).mean()),
        "coupon_disc_active_share": float((tx.COUPON_DISC < 0).mean()),
        "mean_retail_disc_pct_of_regular": float(
            (-tx.RETAIL_DISC / (tx.SALES_VALUE - tx.RETAIL_DISC - tx.COUPON_MATCH_DISC))
            [tx.RETAIL_DISC < 0].mean()),
        "mean_coupon_match_pct_of_loyalty": float(
            (-tx.COUPON_MATCH_DISC / (tx.SALES_VALUE - tx.COUPON_MATCH_DISC))
            [tx.COUPON_MATCH_DISC < 0].mean()),
        "mean_coupon_disc_pct_of_paid": float(
            (-tx.COUPON_DISC / tx.SALES_VALUE)[tx.COUPON_DISC < 0].mean()),
    }
    r["discount_activity"] = act
    log(f"retail_disc active on {act['retail_disc_active_share']:.3f} of lines "
        f"(mean {act['mean_retail_disc_pct_of_regular']:.1%} off the regular price); "
        f"coupon_match on {act['coupon_match_active_share']:.4f}; "
        f"coupon_disc on {act['coupon_disc_active_share']:.4f}")

    # ---------------------------------------------- decisive test per discount
    # The guide claims SALES_VALUE is already net of RETAIL_DISC and of
    # COUPON_MATCH_DISC.  If so, then on a product-day where only *some* shoppers
    # have a given discount active, the ones who do should sit below the day's modal
    # price under a definition that fails to add the discount back, and on it under
    # a definition that does.  That is directly testable.
    def deviation_test(disc_col, defs_to_compare):
        s = sub.copy()
        s["mode_price"] = s.groupby(["PRODUCT_ID", "DAY"])["B_loyalty_shelf"].transform(
            lambda v: np.round(v, 2).mode().iat[0] if len(v) else np.nan)
        active = s[disc_col] < 0
        # only product-days where the discount is mixed across shoppers
        mix = s.groupby(["PRODUCT_ID", "DAY"])[disc_col].transform(lambda v: (v < 0).mean())
        keep = (mix > 0.05) & (mix < 0.95)
        s, active = s[keep], active[keep]
        res = {"product_day_lines": int(keep.sum()),
               "lines_with_discount": int(active.sum())}
        for k in defs_to_compare:
            d_on = (s.loc[active, k] - s.loc[active, "mode_price"]).mean()
            d_off = (s.loc[~active, k] - s.loc[~active, "mode_price"]).mean()
            res[k] = {"mean_gap_to_modal_price_when_discount_active": float(d_on),
                      "mean_gap_when_not_active": float(d_off),
                      "differential": float(d_on - d_off)}
        return res

    r["coupon_match_test"] = deviation_test("COUPON_MATCH_DISC",
                                            ["A_sales_value", "B_loyalty_shelf"])
    r["coupon_disc_test"] = deviation_test("COUPON_DISC",
                                           ["B_loyalty_shelf", "D_customer_paid"])
    for name, key in [("COUPON_MATCH_DISC", "coupon_match_test"),
                      ("COUPON_DISC", "coupon_disc_test")]:
        t = r[key]
        log(f"{name}: on mixed product-days ({t['lines_with_discount']:,} discounted lines "
            f"of {t['product_day_lines']:,}) the differential gap to the modal price is "
            + ", ".join(f"{k} {v['differential']:+.4f}"
                        for k, v in t.items() if isinstance(v, dict)))

    # Does RETAIL_DISC differ across shoppers buying one unit on the same day?  If it
    # were a uniform shelf promotion it would not, and the regular-price definition
    # would be just as concentrated as the loyalty one.
    rd = sub.groupby(["PRODUCT_ID", "DAY"]).RETAIL_DISC.apply(
        lambda v: float(modal_share(v)))
    r["retail_disc_uniform_within_product_day"] = {
        "mean_modal_share": float(rd.mean()),
        "share_of_product_days_fully_uniform": float((rd > 0.999).mean()),
    }
    log(f"RETAIL_DISC is identical for all single-unit buyers on "
        f"{r['retail_disc_uniform_within_product_day']['share_of_product_days_fully_uniform']:.3f} "
        f"of product-days (mean modal share "
        f"{r['retail_disc_uniform_within_product_day']['mean_modal_share']:.3f})")

    # Is the residual dispersion cross-store or cross-shopper?  If the loyalty price
    # is a posted price that simply differs between stores, then conditioning on the
    # store should make it uniform.  Whatever is left after that is genuine
    # shopper-level variation and is the part the chain-level session price averages
    # over.
    # Store-days are too thin to measure this, so use product x store x week and
    # compare against exactly the same transactions pooled across stores -- otherwise
    # the two modal shares are computed on different samples and not comparable.
    # Restrict to items that actually carry a posted price -- random-weight produce
    # and service-counter meat have a continuous "unit price" by construction and
    # would swamp the comparison (see 02_select_sample.price_discreteness).
    posted = set(pd.read_csv(os.path.join(HERE, "..", "model_input", "id_maps", "items.csv"))
                 .PRODUCT_ID) if os.path.exists(
        os.path.join(HERE, "..", "model_input", "id_maps", "items.csv")) else set()
    ws_pool = one[one.PRODUCT_ID.isin(posted)] if posted else one
    n_store_week = ws_pool.groupby(["PRODUCT_ID", "WEEK_NO", "STORE_ID"]).PRODUCT_ID.transform("size")
    ws_lines = ws_pool[n_store_week >= 5]
    # keep product-weeks observed in at least two such stores, so the paired
    # comparison is between the same transactions grouped two ways
    n_stores = ws_lines.groupby(["PRODUCT_ID", "WEEK_NO"]).STORE_ID.transform("nunique")
    ws_lines = ws_lines[n_stores >= 2]
    if len(ws_lines):
        pairs = []
        for (pid, wk), g in ws_lines.groupby(["PRODUCT_ID", "WEEK_NO"]):
            w = g.groupby("STORE_ID")["B_loyalty_shelf"].apply(modal_share)
            n = g.groupby("STORE_ID").size()
            pairs.append({"within": float((w * n).sum() / n.sum()),
                          "pooled": float(modal_share(g["B_loyalty_shelf"])),
                          "n": int(len(g)), "stores": int(len(w))})
        pr = pd.DataFrame(pairs)
        r["within_store_uniformity"] = {
            "product_weeks": int(len(pr)), "lines": int(pr.n.sum()),
            "mean_stores_per_cell": float(pr.stores.mean()),
            "modal_share_within_store": float(np.average(pr.within, weights=pr.n)),
            "modal_share_pooled_across_stores": float(np.average(pr.pooled, weights=pr.n)),
        }
        u = r["within_store_uniformity"]
        log(f"posted-price items, {u['lines']:,} lines in {u['product_weeks']:,} product-weeks "
            f"spanning {u['mean_stores_per_cell']:.1f} stores each: price B modal share "
            f"{u['modal_share_within_store']:.3f} within store vs "
            f"{u['modal_share_pooled_across_stores']:.3f} pooled across stores")

    # Anomalous positive discounts (returns / corrections)
    pos = (tx.RETAIL_DISC > 0) | (tx.COUPON_DISC > 0) | (tx.COUPON_MATCH_DISC > 0)
    r["positive_discount_lines"] = int(pos.sum())
    log(f"lines with a positive (sign-anomalous) discount: {int(pos.sum()):,}")

    # ------------------------------------------------ dispersion around the median
    # Second view: how far is a single transaction from that product-day's median?
    disp = {}
    for k in DEFS:
        med = sub.groupby(["PRODUCT_ID", "DAY"])[k].transform("median")
        disp[k] = float((sub[k] - med).abs().mean())
        log(f"  {k:>16}: mean |price - product-day median| = ${disp[k]:.4f}")
    r["mean_abs_dev_from_daily_median"] = disp

    with open(os.path.join(OUT, "price_audit.json"), "w") as f:
        json.dump(r, f, indent=2)

    # ------------------------------------------------------------------- figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    keys = list(DEFS)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    ax = axes[0]
    vals = [conc[k] for k in keys]
    colors = ["#9aa5b1" if k != "B_loyalty_shelf" else "#2d6cdf" for k in keys]
    bars = ax.barh(range(len(keys)), vals, color=colors)
    ax.set_yticks(range(len(keys)))
    ax.set_yticklabels([LABELS[k] for k in keys], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("mean share of buyers at the modal cent value")
    ax.set_title("Which construction behaves like a posted shelf price?\n"
                 "single-unit purchases, product-days with $\\geq$8 buyers", fontsize=10)
    ax.set_xlim(0, 1)
    for b, v in zip(bars, vals):
        ax.text(v + 0.012, b.get_y() + b.get_height() / 2, f"{v:.3f}",
                va="center", fontsize=9)
    ax.grid(axis="x", alpha=0.3)

    ax = axes[1]
    vals = [disp[k] for k in keys]
    bars = ax.barh(range(len(keys)), vals, color=colors)
    ax.set_yticks(range(len(keys)))
    ax.set_yticklabels([])
    ax.invert_yaxis()
    ax.set_xlabel("mean |unit price $-$ product-day median|  (\\$)")
    ax.set_title("Dispersion among shoppers facing the same shelf", fontsize=10)
    for b, v in zip(bars, vals):
        ax.text(v + max(vals) * 0.02, b.get_y() + b.get_height() / 2, f"${v:.3f}",
                va="center", fontsize=9)
    ax.grid(axis="x", alpha=0.3)

    fig.suptitle("Price reconstruction: all three discount columns, four candidates",
                 fontsize=12, y=1.0)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "price_definitions.png"), dpi=150, bbox_inches="tight")
    log(f"wrote figures/price_definitions.png and out/price_audit.json")


if __name__ == "__main__":
    main()
