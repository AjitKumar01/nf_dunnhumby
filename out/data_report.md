# dunnhumby sample for Nested Factorization

## Raw panel

| index | value |
|---|---|
| lines | 2,553,406 |
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
| category_purchases | 66,637 |
| purchase_rate_per_category_trip | 0.02393 |
| split | {'train': 46432, 'test': 13736, 'validation': 6469} |

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
| own_price_change | 7,954 |
| cross_price_change | 32,382 |
| mean_abs_dp_when_changed | 0.7176 |

## Retained categories

| COMMODITY_DESC | n_items_all | trips_top | multi_any | mean_abs_price_corr | max_share_big |
|---|---|---|---|---|---|
| APPLES | 83 | 967 | 0.047 | 0.321 | 0.233 |
| BABY HBC | 234 | 267 | 0.142 | 0.151 | 0.209 |
| BACON | 88 | 1,420 | 0.05 | 0.158 | 0.547 |
| BATH TISSUES | 125 | 1,858 | 0.039 | 0.213 | 0.244 |
| BATTERIES | 102 | 279 | 0.121 | 0.253 | 0.221 |
| BEEF | 341 | 1,713 | 0.06 | 0.175 | 0.593 |
| BEERS/ALES | 444 | 845 | 0.119 | 0.257 | 0.372 |
| BERRIES | 58 | 2,449 | 0.12 | 0.247 | 0.605 |
| BLEACH | 75 | 601 | 0.063 | 0.288 | 0.291 |
| BREAD | 206 | 869 | 0.088 | 0.218 | 0.267 |
| BREAKFAST SWEETS | 138 | 1,226 | 0.051 | 0.136 | 0.291 |
| BUTTER | 35 | 1,185 | 0.049 | 0.214 | 0.628 |
| CANNED MILK | 33 | 349 | 0.071 | 0.364 | 0.116 |
| CARROTS | 31 | 2,097 | 0.01 | 0.182 | 0.57 |
| CAT LITTER | 82 | 323 | 0.091 | 0.235 | 0.326 |
| CHICKEN/POULTRY | 78 | 979 | 0.06 | 0.186 | 0.628 |
| CITRUS | 91 | 2,127 | 0.119 | 0.206 | 0.209 |
| COCOA MIXES | 89 | 556 | 0.096 | 0.246 | 0.128 |
| DISHWASH DETERGENTS | 128 | 547 | 0.104 | 0.259 | 0.302 |
| DRIED FRUIT | 114 | 420 | 0.118 | 0.21 | 0.174 |
| EGGS | 72 | 6,619 | 0.023 | 0.16 | 0.349 |
| FACIAL TISS/DNR NAPKIN | 84 | 865 | 0.129 | 0.349 | 0.267 |
| FLOUR & MEALS | 64 | 609 | 0.061 | 0.195 | 0.198 |
| FROZEN BREAD/DOUGH | 101 | 885 | 0.093 | 0.232 | 0.267 |
| FROZEN PIE/DESSERTS | 166 | 688 | 0.137 | 0.287 | 0.244 |
| HOT CEREAL | 97 | 636 | 0.138 | 0.142 | 0.384 |
| HOT DOGS | 104 | 1,783 | 0.061 | 0.246 | 0.453 |
| INFANT FORMULA | 99 | 261 | 0.081 | 0.299 | 0.36 |
| LAUNDRY ADDITIVES | 161 | 349 | 0.097 | 0.183 | 0.233 |
| LAUNDRY DETERGENTS | 220 | 565 | 0.043 | 0.353 | 0.453 |
| MARGARINES | 103 | 1,812 | 0.062 | 0.19 | 0.337 |
| MEAT - MISC | 118 | 1,523 | 0.114 | 0.209 | 0.512 |
| MUSHROOMS | 42 | 1,272 | 0.04 | 0.38 | 0.384 |
| NEWSPAPER | 36 | 2,053 | 0.044 | 0.278 | 0.395 |
| NUTS | 67 | 305 | 0.068 | 0.249 | 0.279 |
| OLIVES | 60 | 424 | 0.079 | 0.291 | 0.221 |
| ONIONS | 56 | 1,874 | 0.046 | 0.274 | 0.244 |
| ORGANICS FRUIT & VEGETABLES | 172 | 585 | 0.14 | 0.207 | 0.267 |
| PAPER TOWELS | 75 | 1,808 | 0.057 | 0.243 | 0.186 |
| PICKLE/RELISH/PKLD VEG | 199 | 415 | 0.125 | 0.285 | 0.198 |
| POPCORN | 134 | 269 | 0.101 | 0.231 | 0.233 |
| POTATOES | 65 | 3,107 | 0.028 | 0.326 | 0.372 |
| PROCESSED | 210 | 450 | 0.119 | 0.19 | 0.244 |
| ROLLS | 120 | 772 | 0.066 | 0.21 | 0.326 |
| SALAD MIX | 135 | 2,993 | 0.104 | 0.39 | 0.791 |
| SEAFOOD - FROZEN | 248 | 557 | 0.133 | 0.144 | 0.5 |
| SEAFOOD - SHELF STABLE | 137 | 1,221 | 0.129 | 0.14 | 0.36 |
| SHORTENING/OIL | 164 | 1,076 | 0.082 | 0.226 | 0.198 |
| SNACK NUTS | 159 | 297 | 0.129 | 0.16 | 0.267 |
| SUGARS/SWEETNERS | 93 | 1,950 | 0.099 | 0.36 | 0.221 |
| TOMATOES | 41 | 1,492 | 0.016 | 0.288 | 0.407 |
| TROPICAL FRUIT | 39 | 994 | 0.073 | 0.196 | 0.314 |
| VALUE ADDED VEGETABLES | 49 | 430 | 0.077 | 0.19 | 0.244 |
| VEGETABLES - ALL OTHERS | 104 | 2,962 | 0.14 | 0.267 | 0.36 |
| VEGETABLES SALAD | 38 | 2,397 | 0.027 | 0.177 | 0.407 |
| WATER - CARBONATED/FLVRD DRINK | 271 | 2,009 | 0.15 | 0.194 | 0.302 |
