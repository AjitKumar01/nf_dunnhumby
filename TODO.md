# TODO

## Next

- **Build the phi mask by LIFT, not by co-purchase MASS, and rerun.**
  `pairmask.py` ranks products by total co-purchase count, which favours frequent staples:
  the 20 selected are buns, milk and salad vegetables. Frequent items co-occur often but
  not necessarily with the highest *lift*. Selecting by lift should pick more distinctive
  pairs and use the scarce `c` budget on structure that popularity does not already
  capture. Measured for the current mask: model/empirical pair correlation +0.486, model
  `phi'phi` max 0.517 against an empirical lift of 9.65 on the strongest pair.

## Open questions

- `rho_c` is fitted at commodity granularity (280 groups) and boosts ~518 products when it
  fires, costing 22-35% of ranking MRR. Repartition to sub-commodity (758 groups, ~43
  products boosted) and retrain.
- `beta` (per-product price sensitivity) has no relationship to empirical per-product price
  response (corr +0.016) while the aggregate elasticity is well calibrated. Nothing in the
  objective constrains the allocation across products.
- The `c` budget caps complementarity at `phi'phi ~ 0.35` on 20 products. Raising the grid
  radius costs nodes (~30k for grocery strength at Kz=6); fewer products buys strength
  directly. Neither reaches broad complementarity.
