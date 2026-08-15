"""Where does an iteration actually go?  Timed, not guessed."""
import sys, time, numpy as np, torch
sys.path.insert(0, '../v3'); torch.set_default_dtype(torch.float64)
from data import build; from features import Features; from fit import Batcher
from ragged import (RaggedModel, esp_bucketed, poly_mul_trunc, seg_max, log_f_ragged)

D = build(); F = Features(int(D["n_item"]), int(D["n_store"]), 712); Bt = Batcher(D, F, 120)
m = RaggedModel(J=int(D["n_item"]), N=int(D["n_user"]), C=int(D["n_cat"]), K=32, Kz=12,
                nmax=120, R=23, S=int(D["n_store"]), Kp=8)
m.load_state_dict(torch.load('../../out/v3_run11.pt', map_location='cpu')); m.double().eval()
tr = np.flatnonzero(D["trip_split"] == 0)
ix, ctx, lctx, hh, *_ = Bt.make(tr[:24]); m.house, m.ctx = hh, ctx
print(f"batch 24: slots {ix.item.numel():,}  rows {ix.n_rows:,}  Cpad {ix.Cpad}  "
      f"R {m.R}  Kz {m.Kz}")
print(f"slot_deg: sum {int(ix.slot_deg.sum())}  max {int(ix.slot_deg.max())}  "
      f"mean {float(ix.slot_deg.float().mean()):.2f}")

D_ = 16
z = torch.randn(ix.B, D_, m.Kz)
def t(f, n=3):
    with torch.no_grad():
        f(); t0 = time.time()
        for _ in range(n): f()
    return (time.time() - t0) / n

with torch.no_grad():
    phi_i = m.phi[ix.item]
    bt = m.b_flat(ix) - 0.5 * (phi_i ** 2).sum(-1)
    proj = (z[ix.item_trip] * phi_i.unsqueeze(1)).sum(-1)
    logw = (bt.unsqueeze(1) + proj).transpose(0, 1)
    M = seg_max(logw, ix.item_trip, ix.B)
    w = torch.exp(logw - M.index_select(-1, ix.item_trip))
    e = esp_bucketed(w, ix.row_of, ix.n_rows, m.R, ix.row_size, ix.item_pos)
    r = torch.arange(m.R + 1, dtype=w.dtype)
    a = torch.exp(-m.rho_c[ix.row_cat].unsqueeze(-1) * r * (r - 1) / 2.0)
    G = a.unsqueeze(0) * e
    Gp = torch.zeros(D_, ix.B * ix.Cpad, m.R + 1, dtype=w.dtype); Gp[:, :, 0] = 1.0
    Gp = Gp.index_copy(1, ix.flat_slot, G).view(D_, ix.B, ix.Cpad, m.R + 1)

def conv():
    A = Gp[:, :, 0, :ix.slot_deg[0] + 1]
    for c in range(1, ix.Cpad):
        A = poly_mul_trunc(A, Gp[:, :, c, :ix.slot_deg[c] + 1], m.nmax)
    return A

t_b   = t(lambda: m.b_flat(ix))
t_esp = t(lambda: esp_bucketed(w, ix.row_of, ix.n_rows, m.R, ix.row_size, ix.item_pos))
t_cv  = t(conv)
t_lf  = t(lambda: log_f_ragged(m, z, ix, True))
print(f"\n  b_flat                 {t_b*1000:8.1f} ms")
print(f"  esp_bucketed           {t_esp*1000:8.1f} ms")
print(f"  convolution loop       {t_cv*1000:8.1f} ms   ({ix.Cpad-1} sequential poly_mul)")
print(f"  ---- full log_f        {t_lf*1000:8.1f} ms   convolution is "
      f"{t_cv/t_lf:.0%} of it")
