#!/usr/bin/env python3
"""Build the training-only 280-row co-purchase partition used by version4.html.

The partition is a deterministic preprocessing choice.  It is learned only from training
baskets and caps non-residual groups at 128 products, keeping the exact category
convolution tractable.  Products that never join a supported pair share one residual
group; singleton groups would add computation while their pair statistic is identically
zero.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from data import BI, build


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--minimum-pair-count", type=int, default=8)
    parser.add_argument("--maximum-group-size", type=int, default=128)
    parser.add_argument("--output", type=Path,
                        default=Path(BI) / "items_affinity.parquet")
    args = parser.parse_args()

    data = build()
    pointer, line_item = data["line_ptr"], data["line_item"]
    training = np.flatnonzero(data["trip_split"] == 0)
    products = int(data["n_item"])
    incidence = np.zeros(products, dtype=np.int64)
    pairs: dict[tuple[int, int], int] = {}
    for trip in training:
        lo, hi = int(pointer[trip]), int(pointer[trip + 1])
        items = sorted(set(map(int, line_item[lo:hi])))
        incidence[items] += 1
        for left in range(len(items)):
            for right in range(left + 1, len(items)):
                key = (items[left], items[right])
                pairs[key] = pairs.get(key, 0) + 1

    n = len(training)
    edges = []
    for (left, right), count in pairs.items():
        if count < args.minimum_pair_count:
            continue
        lift = (count * n) / max(int(incidence[left]) * int(incidence[right]), 1)
        if lift > 1.0:
            # Count, rather than lift, maximizes retained observed co-purchase mass.
            edges.append((count, left, right))
    edges.sort(reverse=True)

    parent = np.arange(products)
    size = np.ones(products, dtype=np.int64)

    def find(item: int) -> int:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = int(parent[item])
        return item

    for _count, left, right in edges:
        root_left, root_right = find(left), find(right)
        if root_left == root_right:
            continue
        if size[root_left] + size[root_right] > args.maximum_group_size:
            continue
        if size[root_left] < size[root_right]:
            root_left, root_right = root_right, root_left
        parent[root_right] = root_left
        size[root_left] += size[root_right]

    root = np.asarray([find(item) for item in range(products)])
    root_size = np.bincount(root, minlength=products)
    root[root_size[root] == 1] = -1
    _unique, category = np.unique(root, return_inverse=True)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"item_id": np.arange(products),
                  "cat_id": category}).to_parquet(output, index=False)
    print(f"[affinity] wrote {output}: {products} products, "
          f"{int(category.max()) + 1} groups")
    if products == 5455 and int(category.max()) + 1 != 280:
        raise RuntimeError(
            "the reference data should produce exactly 280 groups; refusing a "
            "silently different model support")


if __name__ == "__main__":
    main()
