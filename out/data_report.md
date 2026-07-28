# dunnhumby sample for Nested Factorization

## Raw panel

| index | value |
|---|---|
| lines | 2,553,408 |
| households | 2,500 |
| products | 91,856 |
| stores | 561 |
| days | 711 |
| weeks | 102 |
| trips | 213,961 |

## Identification: where do prices change?

- consecutive days **within** a `WEEK_NO`: P(price move > 2c) = 0.263 (n=18,476)
- consecutive days **across** the Sunday→Monday week boundary: 0.519 (n=3,357)
- median cross-store CV of a product's price within a week: 0.0109 (31,281 product-weeks with >=5 stores)

## Retained estimation sample

| index | value |
|---|---|
| households | 2,084 |
| items | 560 |
| categories | 56 |
| sessions | 172 |
| pair_weeks | 86 |
| trips | 49,729 |
| category_purchases | 66,638 |
| purchase_rate_per_category_trip | 0.02393 |
| split | {'train': 46431, 'test': 13736, 'validation': 6471} |

## Category filters (paper app. 8.1)

| index | value |
|---|---|
| fewer than 5 items or <200 category-trips | 137 |
| unit demand violated (>15% multi-item or >10% multi-top-item trips) | 79 |
| KEPT | 56 |
| top 15% most seasonal (Herfindahl of daily demand) | 10 |
| insufficient price variation (<2 items changing, or no item with >=10% of pair-weeks moving >=$0.10) | 3 |

## Quasi-experimental variation available

| index | value |
|---|---|
| item_pairweeks | 48,160 |
| own_price_change | 8,087 |
| cross_price_change | 32,446 |
| mean_abs_dp_when_changed | 0.6978 |

## Retained categories

| COMMODITY_DESC | n_items_all | trips_top | multi_any | mean_abs_price_corr | max_share_big |
|---|---|---|---|---|---|
| APPLES | 83 | 967 | 0.047 | 0.331 | 0.244 |
| BABY HBC | 234 | 267 | 0.142 | 0.155 | 0.209 |
| BACON | 88 | 1,421 | 0.05 | 0.163 | 0.57 |
| BATH TISSUES | 125 | 1,858 | 0.039 | 0.215 | 0.267 |
| BATTERIES | 102 | 279 | 0.121 | 0.246 | 0.221 |
| BEEF | 341 | 1,713 | 0.06 | 0.178 | 0.57 |
| BEERS/ALES | 444 | 845 | 0.119 | 0.268 | 0.488 |
| BERRIES | 58 | 2,449 | 0.12 | 0.253 | 0.605 |
| BLEACH | 75 | 601 | 0.063 | 0.303 | 0.291 |
| BREAD | 206 | 869 | 0.088 | 0.198 | 0.244 |
| BREAKFAST SWEETS | 138 | 1,226 | 0.051 | 0.131 | 0.349 |
| BUTTER | 35 | 1,185 | 0.049 | 0.23 | 0.651 |
| CANNED MILK | 33 | 349 | 0.071 | 0.366 | 0.105 |
| CARROTS | 31 | 2,097 | 0.01 | 0.186 | 0.57 |
| CAT LITTER | 82 | 323 | 0.091 | 0.236 | 0.326 |
| CHICKEN/POULTRY | 78 | 979 | 0.06 | 0.223 | 0.651 |
| CITRUS | 91 | 2,127 | 0.119 | 0.204 | 0.221 |
| COCOA MIXES | 89 | 556 | 0.096 | 0.247 | 0.163 |
| DISHWASH DETERGENTS | 128 | 547 | 0.104 | 0.272 | 0.326 |
| DRIED FRUIT | 114 | 420 | 0.118 | 0.212 | 0.14 |
| EGGS | 72 | 6,619 | 0.023 | 0.153 | 0.349 |
| FACIAL TISS/DNR NAPKIN | 84 | 865 | 0.129 | 0.356 | 0.267 |
| FLOUR & MEALS | 64 | 609 | 0.061 | 0.178 | 0.198 |
| FROZEN BREAD/DOUGH | 101 | 885 | 0.093 | 0.232 | 0.221 |
| FROZEN PIE/DESSERTS | 166 | 688 | 0.137 | 0.3 | 0.267 |
| HOT CEREAL | 97 | 636 | 0.138 | 0.165 | 0.349 |
| HOT DOGS | 104 | 1,783 | 0.061 | 0.254 | 0.488 |
| INFANT FORMULA | 99 | 261 | 0.081 | 0.306 | 0.465 |
| LAUNDRY ADDITIVES | 161 | 349 | 0.097 | 0.179 | 0.233 |
| LAUNDRY DETERGENTS | 220 | 565 | 0.043 | 0.381 | 0.384 |
| MARGARINES | 103 | 1,812 | 0.062 | 0.213 | 0.267 |
| MEAT - MISC | 118 | 1,523 | 0.114 | 0.212 | 0.512 |
| MUSHROOMS | 42 | 1,272 | 0.04 | 0.374 | 0.384 |
| NEWSPAPER | 36 | 2,053 | 0.044 | 0.256 | 0.43 |
| NUTS | 67 | 305 | 0.068 | 0.251 | 0.244 |
| OLIVES | 60 | 424 | 0.079 | 0.294 | 0.233 |
| ONIONS | 56 | 1,874 | 0.046 | 0.276 | 0.244 |
| ORGANICS FRUIT & VEGETABLES | 172 | 585 | 0.14 | 0.214 | 0.267 |
| PAPER TOWELS | 75 | 1,808 | 0.057 | 0.239 | 0.198 |
| PICKLE/RELISH/PKLD VEG | 199 | 415 | 0.125 | 0.285 | 0.198 |
| POPCORN | 134 | 269 | 0.101 | 0.241 | 0.233 |
| POTATOES | 65 | 3,107 | 0.028 | 0.274 | 0.395 |
| PROCESSED | 210 | 450 | 0.119 | 0.192 | 0.244 |
| ROLLS | 120 | 772 | 0.066 | 0.234 | 0.314 |
| SALAD MIX | 135 | 2,993 | 0.104 | 0.394 | 0.791 |
| SEAFOOD - FROZEN | 248 | 557 | 0.133 | 0.15 | 0.547 |
| SEAFOOD - SHELF STABLE | 137 | 1,221 | 0.129 | 0.143 | 0.36 |
| SHORTENING/OIL | 164 | 1,076 | 0.082 | 0.233 | 0.198 |
| SNACK NUTS | 159 | 297 | 0.129 | 0.161 | 0.279 |
| SUGARS/SWEETNERS | 93 | 1,950 | 0.099 | 0.364 | 0.221 |
| TOMATOES | 41 | 1,492 | 0.016 | 0.326 | 0.407 |
| TROPICAL FRUIT | 39 | 994 | 0.073 | 0.204 | 0.302 |
| VALUE ADDED VEGETABLES | 49 | 430 | 0.077 | 0.212 | 0.244 |
| VEGETABLES - ALL OTHERS | 104 | 2,962 | 0.14 | 0.254 | 0.36 |
| VEGETABLES SALAD | 38 | 2,397 | 0.027 | 0.162 | 0.349 |
| WATER - CARBONATED/FLVRD DRINK | 271 | 2,009 | 0.15 | 0.212 | 0.302 |
