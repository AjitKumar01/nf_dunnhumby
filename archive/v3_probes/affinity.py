"""Re-partition products by CO-PURCHASE affinity instead of merchandising taxonomy.

The model has two mechanisms for dependence.  phi_j'phi_k is approximate: it needs the
Gaussian integral, and section 14 requires lambda_max < 1 for that integral to be estimable.
rho_c is EXACT -- the category convolution computes it with no draws and no stability
condition -- and rho_c < 0 gives a within-category pair a lift of exp(-rho_c).

Measured today: real baskets need a 2.5x lift on their commonest pairs.  The phi route needs
lambda_max ~ 9 to supply it and the normaliser collapses there (log Z ran 7.4 -> 76.5 as
draws went 8 -> 2048, ESS reading 0.919 throughout).  The rho_c route already reaches 4.48x
on the categories where the taxonomy happens to group complements.

So the mechanism is not missing, it is pointed at the wrong partition: dunnhumby's 188
commodity groups separate pasta from sauce because a merchandiser filed them apart.  This
builds a partition from what shoppers actually buy together, so rho_c can carry the
complementarity that phi cannot.

Method: lift(j,k) = P(j,k) / P(j)P(k) on training baskets, kept where the pair occurs often
enough to be measured; greedy agglomeration on that graph under a size cap, since the
convolution's cost grows with the largest category.
"""
import os, numpy as np, pandas as pd, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
BI = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "basket_input")


def log(m):
    print(f"[aff] {m}", flush=True)


def main(min_pair=8, max_cat=128, out="items_affinity.parquet"):
    from data import build
    D = build()
    lp, li = D["line_ptr"], D["line_item"]
    tr = np.flatnonzero(D["trip_split"] == 0)
    J = int(D["n_item"])
    cnt = np.zeros(J, np.int64)
    pair = {}
    for t in tr:
        a, b = int(lp[t]), int(lp[t + 1])
        s = sorted(set(int(x) for x in li[a:b]))
        for j in s:
            cnt[j] += 1
        for i in range(len(s)):
            for k in range(i + 1, len(s)):
                key = (s[i], s[k])
                pair[key] = pair.get(key, 0) + 1
    n = len(tr)
    log(f"{n:,} training baskets, {len(pair):,} distinct co-purchased pairs")

    edges = []
    for (j, k), c in pair.items():
        if c < min_pair:
            continue
        lift = (c / n) / max((cnt[j] / n) * (cnt[k] / n), 1e-12)
        if lift > 1.0:
            # Order by co-purchase COUNT, not lift.  The objective is to put as much
            # observed co-purchase MASS as possible inside a group, and the highest-lift
            # pairs are the rarest ones -- ordering by lift merged obscure pairs first and
            # captured 3.8% of the mass, below the 4.1% the commodity taxonomy already gets.
            edges.append((c, j, k))
    edges.sort(reverse=True)
    log(f"{len(edges):,} pairs with count >= {min_pair} and lift > 1; "
        f"top count {edges[0][0]}, median {edges[len(edges)//2][0]}")

    # greedy agglomeration, strongest pair first, capped so the convolution stays cheap
    lab = np.arange(J)
    size = np.ones(J, np.int64)
    def find(x):
        while lab[x] != x:
            lab[x] = lab[lab[x]]; x = lab[x]
        return x
    merged = 0
    for lift, j, k in edges:
        rj, rk = find(j), find(k)
        if rj == rk or size[rj] + size[rk] > max_cat:
            continue
        if size[rj] < size[rk]:
            rj, rk = rk, rj
        lab[rk] = rj; size[rj] += size[rk]; merged += 1
    root = np.array([find(j) for j in range(J)])
    # Products that joined no group go into ONE shared bucket rather than 1,774 singleton
    # categories.  A one-product category contributes a trivial polynomial and still costs a
    # full row in the convolution: singletons drove rows from 4,076 to 25,738 and Cpad from
    # 183 to 1,649, at 3.25 s/iteration.  Their rho_c has nothing to act on either way, so
    # collapsing them is free in modelling terms and large in compute terms.
    rsize = np.bincount(root, minlength=J)
    lone = rsize[root] == 1
    root = np.where(lone, -1, root)
    uniq, newcat = np.unique(root, return_inverse=True)
    sizes = np.bincount(newcat)
    log(f"{merged:,} merges -> {len(uniq):,} affinity groups "
        f"(was {int(D['n_cat'])} commodity categories)")
    log(f"group size: median {int(np.median(sizes))}  mean {sizes.mean():.1f}  "
        f"max {sizes.max()}  singletons {int((sizes==1).sum()):,}")

    # how much of the observed co-purchase mass now falls INSIDE a group?
    tot = win = 0
    for (j, k), c in pair.items():
        tot += c
        if newcat[j] == newcat[k]:
            win += c
    old = pd.read_parquet(os.path.join(BI, "items.parquet"))[["item_id", "cat_id"]]
    o = old.set_index("item_id").cat_id.reindex(range(J)).to_numpy()
    tot_o = win_o = 0
    for (j, k), c in pair.items():
        tot_o += c
        if o[j] == o[k]:
            win_o += c
    log(f"co-purchase mass inside a group: affinity {win/tot:.1%}  "
        f"vs commodity {win_o/tot_o:.1%}")
    pd.DataFrame({"item_id": np.arange(J), "cat_id": newcat}).to_parquet(
        os.path.join(BI, out))
    log(f"wrote {out}")


if __name__ == "__main__":
    main()
