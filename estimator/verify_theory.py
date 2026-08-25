"""Verify EVERY step of the derivation numerically before implementing anything.

  (1) closed form for E[exp(||v||^2/2)] under v ~ N(a,B)          -- pure maths
  (2) Z = f(0) * E_{P_W}[exp(||v_S||^2/2)]                        -- by enumeration
  (3) a = grad log f(0) = E[v_S] and B = Hess = Cov(v_S)          -- by enumeration
  (4) the tilted-IS identity is UNBIASED (converges to exact)     -- node sweep
  (5) unbiasedness holds even with a deliberately WRONG B         -- the identity claim
"""
import math, itertools, sys
import numpy as np, torch
sys.path.insert(0, '.')
torch.set_default_dtype(torch.float64)
import gauss_est as G
from sim_test import exact
from sim_test2 import case

ok = lambda c: "PASS" if c else "**FAIL**"

# ---- (1) closed form vs brute-force Monte Carlo on a known Gaussian --------------
d = 4
g = torch.Generator().manual_seed(0)
Braw = torch.randn(d, d, generator=g, dtype=torch.float64) * 0.3
B = Braw @ Braw.T
B = B / (torch.linalg.eigvalsh(B).max() / 0.55)         # lambda_max = 0.55 < 1
a = torch.randn(d, generator=g, dtype=torch.float64) * 0.3
I = torch.eye(d, dtype=torch.float64)
closed = -0.5 * torch.logdet(I - B) + 0.5 * a @ torch.linalg.solve(I - B, a)
L = torch.linalg.cholesky(B)
X = a.unsqueeze(0) + torch.randn(4_000_000, d, generator=g, dtype=torch.float64) @ L.T
mc = torch.logsumexp(0.5 * (X * X).sum(-1), 0) - math.log(X.shape[0])
print(f"(1) E[exp(||v||^2/2)], v ~ N(a,B), lam_max(B)=0.55")
print(f"    closed form {float(closed):.6f}   MC(4e6) {float(mc):.6f}   "
      f"diff {float(closed-mc):+.2e}   {ok(abs(float(closed-mc))<3e-3)}")

# ---- (2)(3) enumeration checks on a small model ---------------------------------
J, Kz, nmax = 12, 6, 12
b, phi, cat, rc, r0 = case(J, Kz, 0.7, nmax, -0.5, seed=3)
subs = [list(S) for n in range(1, J + 1) for S in itertools.combinations(range(J), n)]
logW = torch.tensor([float(b[S].sum()) - 0.5 * float((phi[S] ** 2).sum())
                     - float((rc * torch.bincount(cat[S], minlength=len(rc)).double()
                              * (torch.bincount(cat[S], minlength=len(rc)).double() - 1) / 2).sum())
                     - float(r0[len(S)]) for S in subs])
V = torch.stack([phi[S].sum(0) for S in subs])
p = torch.softmax(logW, 0)
f0_enum = torch.logsumexp(logW, 0)
z0 = torch.zeros(Kz, requires_grad=True)
f0_code = G.log_f(z0, b, phi, cat, rc, r0, nmax)
a_code = torch.autograd.grad(f0_code, z0, create_graph=True)[0]
Bc = torch.stack([torch.autograd.grad(a_code[k], z0, retain_graph=True)[0] for k in range(Kz)])
a_enum = (p.unsqueeze(-1) * V).sum(0)
B_enum = (p.unsqueeze(-1).unsqueeze(-1) * torch.einsum('si,sj->sij', V - a_enum, V - a_enum)).sum(0)
ex = exact(b, phi, cat, rc, r0, nmax)
step2 = float(f0_enum) + float(torch.logsumexp(logW + 0.5 * (V * V).sum(-1), 0) - f0_enum)
print(f"\n(2) Z = f(0) * E_P[exp(||v||^2/2)]")
print(f"    enumerated {step2:.8f}   exact {ex:.8f}   diff {step2-ex:+.2e}   {ok(abs(step2-ex)<1e-8)}")
print(f"\n(3) a = grad log f(0) = E[v] ; B = Hess = Cov(v)")
print(f"    max|a_autograd - a_enum| {float((a_code-a_enum).abs().max()):.2e}   "
      f"{ok(float((a_code-a_enum).abs().max())<1e-9)}")
print(f"    max|B_autograd - B_enum| {float((Bc-B_enum).abs().max()):.2e}   "
      f"{ok(float((Bc-B_enum).abs().max())<1e-9)}")

# ---- (4) tilted IS converges to exact --------------------------------------------
print(f"\n(4) tilted-IS convergence (exact = {ex:.5f})")
for n in (4, 16, 64, 256, 1024):
    vals = [G.log_Z_cv(b, phi, cat, rc, r0, nmax, n, seed=1000*s, reps=2) for s in range(6)]
    print(f"    {n:>5} nodes: mean {np.mean(vals)-ex:+.6f}   sd {np.std(vals):.6f}")

# ---- (5) unbiased even with a WRONG B --------------------------------------------
print(f"\n(5) identity claim: unbiased for ANY valid B_eff (not just the true Cov)")
orig = G.log_Z_cv
import types
def cv_wrongB(scale, n, seed):
    """Force B_eff = scale * B_mf to prove the identity does not depend on B being right."""
    src = orig.__code__
    Kz = phi.shape[1]
    z0 = torch.zeros(Kz, requires_grad=True)
    g0 = G.log_f(z0, b, phi, cat, rc, r0, nmax)
    av = torch.autograd.grad(g0, z0)[0].detach(); g0 = g0.detach()
    bb = b.detach().clone().requires_grad_(True)
    pi = torch.autograd.grad(G.log_f(torch.zeros(Kz), bb, phi, cat, rc, r0, nmax), bb)[0]
    Bq = scale * (phi.T @ (phi * (pi * (1 - pi)).unsqueeze(-1)))
    Ie = torch.eye(Kz, dtype=torch.float64); M = Ie - Bq
    if float(torch.linalg.eigvalsh(M).min()) <= 1e-6: return float('nan')
    Vv = torch.linalg.inv(M); m = Vv @ av; L = torch.linalg.cholesky(Vv)
    lzq = g0 - 0.5*torch.logdet(M) + 0.5*(av @ (Vv @ av))
    eng = torch.quasirandom.SobolEngine(Kz, scramble=True, seed=seed)
    u = eng.draw(n).double().clamp(1e-12, 1-1e-12)
    x = torch.erfinv(2*u-1)*math.sqrt(2.0)
    lr_ = []
    for i in range(n):
        z = m + L @ x[i]
        lr_.append(G.log_f(z, b, phi, cat, rc, r0, nmax) - (g0 + av@z + 0.5*(z@(Bq@z))))
    return float(lzq + torch.logsumexp(torch.stack(lr_),0) - math.log(n))
for sc in (0.0, 0.5, 1.0, 1.5):
    vals = [cv_wrongB(sc, 512, 1000*s) for s in range(4)]
    print(f"    B_eff = {sc:.1f} x B_mf : mean {np.mean(vals)-ex:+.6f}  sd {np.std(vals):.6f}")
