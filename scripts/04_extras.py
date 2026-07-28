"""
Stage 4 -- Build the dunnhumby-only signals the paper's scanner data did not carry.

The paper's data had prices and stock-outs.  dunnhumby adds three things that map
naturally onto the Nested Factorization utility:

1. causal_data: in-store display and weekly-mailer feature at product x store x week.
   This is a *time-varying item attribute*, exactly the slot that price occupies in
   equation (4).  It enters as an extra bilinear term  (delta_i . omega_j) * display_jt,
   so promotion response is heterogeneous across households the same way price
   sensitivity is.  Because bemb_loc's session prices must be common to all users,
   the panel is collapsed to chain level, traffic-weighted by store.

2. coupon / campaign_table / campaign_desc: which household was eligible for which
   product's coupon, and when.  This is genuine *household x item x time* variation --
   something the paper had no analogue for, and the thing its sec. 6.5 targeting
   counterfactual could only simulate.  Encoded as an eligibility weight:
     TypeB/TypeC -- every participating household gets every coupon -> weight 1
     TypeA       -- each household receives 16 coupons drawn from the campaign pool
                    and which 16 is not recorded, so the weight is 16/pool_size.

3. coupon_redempt: realised redemptions, held back from training and used purely as
   a validation target for targeted-discount counterfactuals.

Outputs (in ../model_input):
    item_sess_display.tsv   item_id \t session_id \t display_intensity
    item_sess_mailer.tsv    item_id \t session_id \t mailer_intensity
    user_item_sess_coupon.tsv  user_id \t item_id \t session_id \t weight  (sparse)
    redemptions.csv         household/product/day redemptions inside the sample
"""
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
MI = os.path.join(HERE, "..", "model_input")
# Raw dunnhumby CSVs.  Defaults to a sibling of the repository; override with
# NF_RAW_DIR if the download lives somewhere else.
RAW = os.path.join(os.environ.get(
    "NF_RAW_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                 "dunnhumby_The-Complete-Journey",
                 "dunnhumby_The-Complete-Journey CSV")), "")

MAILER_FEATURE = set("ACDFHJLPXZ")   # anything other than '0' is some kind of ad


def log(m):
    print(f"[04] {m}", flush=True)


def main():
    items = pd.read_csv(os.path.join(MI, "id_maps", "items.csv"))
    sess = pd.read_csv(os.path.join(MI, "id_maps", "sessions.csv"))
    users = pd.read_csv(os.path.join(MI, "id_maps", "users.csv"))
    keep_prod = set(items.PRODUCT_ID)
    iid = items.set_index("PRODUCT_ID").item_id
    uid = users.set_index("household_key").user_id

    # ------------------------------------------------------- store traffic weights
    tx = pd.read_parquet(os.path.join(DATA, "tx.parquet"), columns=["STORE_ID", "BASKET_ID"])
    traffic = tx.groupby("STORE_ID").BASKET_ID.nunique().rename("w")
    traffic = traffic / traffic.sum()

    # --------------------------------------------------------------- causal_data
    log("streaming causal_data.csv (36.8M rows) ...")
    chunks = []
    for ch in pd.read_csv(RAW + "causal_data.csv", chunksize=2_000_000,
                          dtype={"PRODUCT_ID": np.int32, "STORE_ID": np.int32,
                                 "WEEK_NO": np.int16, "display": "string", "mailer": "string"}):
        ch = ch[ch.PRODUCT_ID.isin(keep_prod)]
        if len(ch):
            chunks.append(ch)
    cz = pd.concat(chunks, ignore_index=True)
    log(f"causal_data rows for retained items: {len(cz):,}")
    cz["disp"] = (cz.display.fillna("0").str.strip() != "0").astype(float)
    cz["mail"] = cz.mailer.fillna("0").str.strip().isin(MAILER_FEATURE).astype(float)
    cz["w"] = cz.STORE_ID.map(traffic).fillna(0.0)

    agg = cz.groupby(["PRODUCT_ID", "WEEK_NO"]).apply(
        lambda d: pd.Series({"display": float((d.disp * d.w).sum()),
                             "mailer": float((d.mail * d.w).sum())}), include_groups=False
    ).reset_index()
    # causal_data only lists product-store-weeks with *some* promotion, so the
    # traffic-weighted sum is already the share of shopper traffic exposed to it.
    agg["display"] = agg.display.clip(0, 1)
    agg["mailer"] = agg.mailer.clip(0, 1)
    log(f"promoted item-weeks: {len(agg):,}  mean display {agg.display.mean():.3f}  "
        f"mean mailer {agg.mailer.mean():.3f}")

    grid = pd.MultiIndex.from_product([items.item_id.values, sess.session_id.values],
                                      names=["item_id", "session_id"]).to_frame(index=False)
    agg["item_id"] = agg.PRODUCT_ID.map(iid)
    promo = grid.merge(sess[["session_id", "WEEK_NO"]], on="session_id") \
                .merge(agg[["item_id", "WEEK_NO", "display", "mailer"]],
                       on=["item_id", "WEEK_NO"], how="left").fillna({"display": 0.0, "mailer": 0.0})

    for col, fname in [("display", "item_sess_display.tsv"), ("mailer", "item_sess_mailer.tsv")]:
        with open(os.path.join(MI, fname), "w") as f:
            for r in promo[["item_id", "session_id", col]].itertuples(index=False):
                f.write(f"{int(r.item_id)}\t{int(r.session_id)}\t{getattr(r, col):.4f}\n")
        share = (promo[col] > 0).mean()
        log(f"wrote {fname} ({len(promo):,} lines);  {share:.3f} of item-sessions promoted")

    # ------------------------------------------------------------------- coupons
    cp = pd.read_csv(RAW + "coupon.csv")
    ct = pd.read_csv(RAW + "campaign_table.csv")
    cd = pd.read_csv(RAW + "campaign_desc.csv")
    cp = cp[cp.PRODUCT_ID.isin(keep_prod)]
    log(f"coupon rows covering retained items: {len(cp):,} over {cp.CAMPAIGN.nunique()} campaigns")

    pool = cp.groupby("CAMPAIGN").COUPON_UPC.nunique().rename("pool")
    cd = cd.merge(pool, left_on="CAMPAIGN", right_index=True, how="left")
    # probability a participating household holds a given coupon from the campaign
    cd["p_hold"] = np.where(cd.DESCRIPTION.str.upper() == "TYPEA",
                            np.minimum(1.0, 16.0 / cd.pool.fillna(16)), 1.0)

    hh_camp = ct.merge(cd[["CAMPAIGN", "START_DAY", "END_DAY", "p_hold"]], on="CAMPAIGN", how="inner")
    hh_camp = hh_camp[hh_camp.household_key.isin(set(users.household_key))]
    cp_prod = cp[["CAMPAIGN", "PRODUCT_ID"]].drop_duplicates()

    elig = hh_camp.merge(cp_prod, on="CAMPAIGN")
    log(f"household x product x campaign eligibility rows: {len(elig):,}")

    # Store eligibility in interval form (user, item, first_session, last_session,
    # weight).  Expanding it cell by cell would be ~14M lines for no extra
    # information: eligibility is constant within a campaign window.
    s = sess[["session_id", "DAY"]].sort_values("DAY").reset_index(drop=True)
    rows = []
    for (start, end), g in elig.groupby(["START_DAY", "END_DAY"]):
        days = s[(s.DAY >= start) & (s.DAY <= end)]
        if not len(days):
            continue
        lo, hi = int(days.session_id.min()), int(days.session_id.max())
        gg = g[["household_key", "PRODUCT_ID", "p_hold"]].copy()
        gg["s_lo"], gg["s_hi"] = lo, hi
        rows.append(gg)
    if rows:
        ce = pd.concat(rows, ignore_index=True)
        ce["user_id"] = ce.household_key.map(uid)
        ce["item_id"] = ce.PRODUCT_ID.map(iid)
        ce = ce.dropna(subset=["user_id", "item_id"])
        ce = ce.groupby(["user_id", "item_id", "s_lo", "s_hi"], as_index=False).p_hold.max()
        ce[["user_id", "item_id", "s_lo", "s_hi", "p_hold"]].astype(
            {"user_id": int, "item_id": int, "s_lo": int, "s_hi": int}).to_csv(
            os.path.join(MI, "user_item_sess_coupon.tsv"), sep="\t", header=False, index=False,
            float_format="%.4f")
        n_cells = int(((ce.s_hi - ce.s_lo + 1)).sum())
        log(f"wrote user_item_sess_coupon.tsv ({len(ce):,} intervals covering {n_cells:,} "
            f"user-item-session cells; {ce.user_id.nunique()} households, {ce.item_id.nunique()} items)")

        # Campaign-factored form for the model: coupon_ijt = max_k w_k * U[i,k] P[j,k] S[t,k].
        # Eligibility is a product of three memberships, so this is exact and tiny
        # (21 campaigns) where the expanded tensor would be 10^8 cells.
        camps = sorted(set(hh_camp.CAMPAIGN) & set(cp_prod.CAMPAIGN))
        cidx = {c: k for k, c in enumerate(camps)}
        nU, nJ, nS = len(users), len(items), len(sess)
        U = np.zeros((nU, len(camps)), dtype=np.float32)
        P = np.zeros((nJ, len(camps)), dtype=np.float32)
        S = np.zeros((nS, len(camps)), dtype=np.float32)
        wgt = np.zeros(len(camps), dtype=np.float32)
        for c in camps:
            k = cidx[c]
            hu = hh_camp.loc[hh_camp.CAMPAIGN == c, "household_key"].map(uid).dropna()
            U[hu.astype(int).values, k] = 1.0
            pi = cp_prod.loc[cp_prod.CAMPAIGN == c, "PRODUCT_ID"].map(iid).dropna()
            P[pi.astype(int).values, k] = 1.0
            row = cd.loc[cd.CAMPAIGN == c].iloc[0]
            days = sess[(sess.DAY >= row.START_DAY) & (sess.DAY <= row.END_DAY)]
            S[days.session_id.values.astype(int), k] = 1.0
            wgt[k] = float(row.p_hold)
        np.savez(os.path.join(MI, "coupon_campaigns.npz"), U=U, P=P, S=S, w=wgt,
                 campaigns=np.array(camps))
        live = (U.sum(0) > 0) & (P.sum(0) > 0) & (S.sum(0) > 0)
        log(f"wrote coupon_campaigns.npz ({len(camps)} campaigns, {int(live.sum())} active "
            f"inside the retained sessions)")
    else:
        log("no coupon eligibility overlaps the retained sessions")

    # ---------------------------------------------------------------- redemptions
    red = pd.read_csv(RAW + "coupon_redempt.csv")
    red = red.merge(cp[["COUPON_UPC", "PRODUCT_ID"]].drop_duplicates(), on="COUPON_UPC", how="inner")
    red = red[red.household_key.isin(set(users.household_key))]
    red.to_csv(os.path.join(MI, "redemptions.csv"), index=False)
    log(f"wrote redemptions.csv ({len(red):,} household-coupon-product redemptions on retained items)")


if __name__ == "__main__":
    main()
