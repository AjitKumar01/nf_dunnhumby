"""Replace the sequential category convolution with a balanced product tree, and check the
per-slot degree cap is actually capping anything."""
import sys, time, numpy as np, torch
sys.path.insert(0, '../v3'); torch.set_default_dtype(torch.float64)
from data import build; from features import Features; from fit import Batcher
from ragged import RaggedModel, esp_bucketed, poly_mul_trunc, seg_max

D = build(); F = Features(int(D["n_item"]), int(D["n_store"]), 712); Bt = Batcher(D, F, 120)
m = RaggedModel(J=int(D["n_item"]), N=int(D["n_user"]), C=int(D["n_cat"]), K=32, Kz=12,
                nmax=120, R=23, S=int(D["n_store"]), Kp=8)
m.load_state_dict(torch.load('../../out/v3_run11.pt', map_location='cpu')); m.double().eval()
tr = np.flatnonzero(D["trip_split"] == 0)
ix, ctx, lctx, hh, *_ = Bt.make(tr[:24]); m.house, m.ctx = hh, ctx

# --- is the degree cap capping? Gp's last axis is only R+1 long -------------------------
eff = torch.minimum(ix.slot_deg, torch.tensor(m.R)) + 1
print(f"Cpad {ix.Cpad}   R+1 {m.R+1}")
print(f"slot_deg raw: max {int(ix.slot_deg.max())} mean {float(ix.slot_deg.float().mean()):.1f}"
      f"  -- this is the ASSORTMENT category size, not a basket count")
print(f"coefficients actually used {int(eff.sum()):,} of {ix.Cpad*(m.R+1):,} "
      f"({int(eff.sum())/(ix.Cpad*(m.R+1)):.1%})   slots capped below R: "
      f"{int((ix.slot_deg < m.R).sum())}/{ix.Cpad}")

D_ = 16
z = torch.randn(ix.B, D_, m.Kz)
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

def seq():
    A = Gp[:, :, 0, :ix.slot_deg[0] + 1]
    for c in range(1, ix.Cpad):
        A = poly_mul_trunc(A, Gp[:, :, c, :ix.slot_deg[c] + 1], m.nmax)
    return A

def tree(P, nmax):
    """Product of the [..., C, deg] polynomials, pairwise, log2(C) rounds not C."""
    while P.shape[-2] > 1:
        C, d = P.shape[-2], P.shape[-1]
        if C % 2:
            pad = torch.zeros(P.shape[:-2] + (1, d), dtype=P.dtype, device=P.device)
            pad[..., 0] = 1.0
            P = torch.cat([P, pad], dim=-2); C += 1
        A, Bp = P[..., 0::2, :], P[..., 1::2, :]
        nd = min(2 * d - 1, nmax + 1)
        out = torch.zeros(P.shape[:-2] + (C // 2, nd), dtype=P.dtype, device=P.device)
        for k in range(min(d, nd)):
            take = min(d, nd - k)
            out[..., k:k + take] = out[..., k:k + take] + A[..., :take] * Bp[..., k:k + 1]
        P = out
    return P[..., 0, :]

with torch.no_grad():
    a1, a2 = seq(), tree(Gp, m.nmax)
    n = min(a1.shape[-1], a2.shape[-1])
    rel = float(((a1[..., :n] - a2[..., :n]).abs() / a1[..., :n].abs().clamp_min(1e-300)).max())
    print(f"\ntree vs sequential: max relative difference {rel:.3e}")
    def t(f, k=3):
        f(); t0 = time.time()
        for _ in range(k): f()
        return (time.time() - t0) / k
    ts, tt = t(seq), t(lambda: tree(Gp, m.nmax))
    print(f"sequential {ts*1000:8.1f} ms   ({ix.Cpad-1} steps)")
    print(f"tree       {tt*1000:8.1f} ms   ({int(np.ceil(np.log2(ix.Cpad)))} rounds)"
          f"   speedup {ts/tt:.1f}x")
