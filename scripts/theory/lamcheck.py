"""lambda_max is computed through esp_newton.  Newton's identities were measured to fail
above R ~ 4 (log Z error 3.3e+264 at R=23) and log_f_ragged was moved to the stable
bucketed recursion because of it -- but lambda_max was not.  This recomputes it both ways."""
import sys, numpy as np, torch
sys.path.insert(0, '../v3'); torch.set_default_dtype(torch.float64)
from data import build; from features import Features; from fit import Batcher
import ragged
from ragged import RaggedModel, esp_newton, esp_bucketed

D = build(); F = Features(int(D["n_item"]), int(D["n_store"]), 712); Bt = Batcher(D, F, 120)
m = RaggedModel(J=int(D["n_item"]), N=int(D["n_user"]), C=int(D["n_cat"]), K=32, Kz=12,
                nmax=120, R=23, S=int(D["n_store"]), Kp=8)
m.load_state_dict(torch.load('../../out/v3_run11.pt', map_location='cpu')); m.double().eval()
tr = np.flatnonzero(D["trip_split"] == 0)

# direct comparison of the two ESP routines on a real batch
ix, ctx, lctx, hh, *_ = Bt.make(tr[:24]); m.house, m.ctx = hh, ctx
z = torch.zeros(ix.B, 1, m.Kz)
with torch.no_grad():
    phi_i = m.phi[ix.item]
    bt = m.b_flat(ix) - 0.5 * (phi_i ** 2).sum(-1)
    proj = (z[ix.item_trip] * phi_i.unsqueeze(1)).sum(-1)
    logw = (bt.unsqueeze(1) + proj).transpose(0, 1)
    M = ragged.seg_max(logw, ix.item_trip, ix.B)
    w = torch.exp(logw - M.index_select(-1, ix.item_trip))
    en = esp_newton(w, ix.row_of, ix.n_rows, m.R, ix.row_size)
    eb = esp_bucketed(w, ix.row_of, ix.n_rows, m.R, ix.row_size, ix.item_pos)
d = (en - eb).abs()
print(f"esp_newton vs esp_bucketed on a real batch at R={m.R}:")
print(f"  max abs difference {float(d.max()):.4e}   max |bucketed| {float(eb.abs().max()):.4e}")
print(f"  rows where they differ by >1% of the bucketed value: "
      f"{int((d > 0.01*eb.abs()).any(-1).sum())} / {ix.n_rows}")
print(f"  negative coefficients: newton {int((en < 0).sum())}   bucketed {int((eb < 0).sum())}")

# lambda_max as shipped, vs the same function with the stable recursion
vals = {}
for name, fn in (("esp_newton (as shipped)", esp_newton),
                 ("esp_bucketed (stable)", None)):
    if fn is None:
        orig = ragged.esp_newton
        ragged.esp_newton = lambda w, r, n, R, rs: esp_bucketed(w, r, n, R, rs, ix.item_pos)
    out = []
    for i in range(0, 96, 24):
        jx, c2, l2, h2, *_ = Bt.make(tr[i:i+24]); m.house, m.ctx = h2, c2
        out.append(m.lambda_max(jx))
    if fn is None:
        ragged.esp_newton = orig
    vals[name] = out
    print(f"\n{name}:  " + "  ".join(f"{v:.4f}" for v in out))
