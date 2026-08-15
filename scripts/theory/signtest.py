"""Does the constrained model give the right sign for a price cut -- everywhere?

The likelihood comparison says what the constraint cost.  This says whether it worked.  A
10% cut must raise every product's index (d b_j > 0) and must never lower it, for every
household-product pair, not merely for the one product that exposed the bug.
"""
import sys, math, numpy as np, torch
sys.path.insert(0, '../v3'); torch.set_default_dtype(torch.float64)
from torch.nn.functional import softplus
from data import build; from features import Features; from fit import Batcher
from ragged import RaggedModel

D = build(); F = Features(int(D["n_item"]), int(D["n_store"]), 712); Bt = Batcher(D, F, 120)
J, N, C, S = (int(D[k]) for k in ("n_item", "n_user", "n_cat", "n_store"))
dlogp = math.log(0.90)

for tag, ck, constrained in (("run17 (unconstrained)", 'v3_run17.pt', False),
                             ("run18 (softplus)",      'v3_run18.pt', True)):
    m = RaggedModel(J=J, N=N, C=C, K=32, Kz=12, nmax=120, R=23, S=S, Kp=8)
    m.load_state_dict(torch.load(f'../../out/{ck}', map_location='cpu'))
    m.double().eval()
    g, b = m.gamma.detach(), m.beta.detach()
    if constrained:
        g, b = softplus(g), softplus(b)
    # d b_j / d log p_j = -(gamma_h . beta_j); a 10% cut gives db = -(g.b) * log(0.9)
    gb = g @ b.T                                     # [N, J] every household x product
    db = -gb * dlogp
    tot = gb.numel()
    print(f"\n{tag}")
    print(f"  gamma.beta over {tot:,} household-product pairs:")
    print(f"    negative (price rise makes it MORE likely): {int((gb < 0).sum()):,} "
          f"({float((gb < 0).float().mean()):.2%})")
    print(f"    min {float(gb.min()):+.4f}   median {float(gb.median()):+.4f}   "
          f"max {float(gb.max()):+.4f}")
    print(f"  10% cut, d b_j:  min {float(db.min()):+.5f}   "
          f"median {float(db.median()):+.5f}   max {float(db.max()):+.5f}")
    print(f"    pairs where a cut LOWERS the index: {int((db < 0).sum()):,}")
