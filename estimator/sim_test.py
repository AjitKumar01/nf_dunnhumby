"""Simulated-data test of the quadrature-free estimator: ACCURACY and LATENCY.

Ground truth is exact enumeration over all 2^J-1 subsets, so J stays small enough to
enumerate.  Regimes sweep the two things that actually stress the estimator: interaction
strength (lambda_max(Cov v) -> 1 is where the expansion must fail) and basket size.
"""
import math, time, itertools, sys
import numpy as np, torch
sys.path.insert(0, '/Users/ajit/Projects/Causal/nf_dunnhumby/estimator')
torch.set_default_dtype(torch.float64)
from gauss_est import log_f, log_Z_gauss, esp_weighted


def exact(b, phi, cat, rho_c, rho0, nmax):
    J = len(b); terms = []
    for n in range(1, min(J, nmax) + 1):
        for S in itertools.combinations(range(J), n):
            S = list(S)
            v = phi[S].sum(0)
            e = float(b[S].sum()) + 0.5 * float(v @ v - (phi[S] ** 2).sum())
            cc = torch.bincount(cat[S], minlength=len(rho_c)).to(b.dtype)
            e -= float((rho_c * cc * (cc - 1) / 2).sum()) + float(rho0[n])
            terms.append(e)
    return float(torch.logsumexp(torch.tensor(terms), 0))


def qmc(b, phi, cat, rho_c, rho0, nmax, n_nodes, seed=0, reps=4):
    """Mode-shifted scrambled Sobol, the current approach, as the latency/accuracy baseline."""
    Kz = phi.shape[1]
    zc = torch.zeros(Kz, requires_grad=True)
    for _ in range(3):                                   # damped ascent to the mode
        gz = torch.autograd.grad(log_f(zc, b, phi, cat, rho_c, rho0, nmax)
                                 - 0.5 * (zc ** 2).sum(), zc)[0]
        zc = (zc + 0.5 * gz).detach().requires_grad_(True)
    zh = zc.detach()
    acc = []
    for r in range(reps):
        eng = torch.quasirandom.SobolEngine(Kz, scramble=True, seed=seed + 7919 * r)
        u = eng.draw(max(n_nodes // reps, 1)).double().clamp(1e-12, 1 - 1e-12)
        x = torch.erfinv(2 * u - 1) * math.sqrt(2.0)
        lg = []
        for i in range(x.shape[0]):
            zz = zh + x[i]
            lg.append(log_f(zz, b, phi, cat, rho_c, rho0, nmax)
                      - 0.5 * (zz @ zz) + 0.5 * (x[i] @ x[i]))
        acc.append(torch.logsumexp(torch.stack(lg), 0) - math.log(x.shape[0]))
    return float(torch.stack(acc).logsumexp(0) - math.log(reps))


def case(J, Kz, rho, nmax, seed, ncat=4, bmean=-1.2):
    g = torch.Generator().manual_seed(seed)
    phi = torch.randn(J, Kz, generator=g, dtype=torch.float64)
    phi = phi / phi.norm(dim=1, keepdim=True) * rho
    b = torch.randn(J, generator=g, dtype=torch.float64) * 0.4 + bmean
    cat = torch.arange(J) % ncat
    rho_c = torch.full((ncat,), 0.08, dtype=torch.float64)
    rho0 = 0.03 * torch.arange(nmax + 1, dtype=torch.float64) ** 2
    return b, phi, cat, rho_c, rho0


print("Simulated data.  Ground truth = exact enumeration over all 2^J-1 subsets.\n")
print(f"{'J':>3}{'Kz':>4}{'rho':>6}{'lam_max':>9}{'exact':>10}"
      f"{'gauss-exactB':>14}{'ms':>7}{'gauss-mf':>10}{'ms':>7}"
      f"{'QMC-8':>9}{'ms':>7}{'QMC-128':>10}{'ms':>7}")
for J, Kz, rho in ((14, 8, 0.30), (14, 8, 0.60), (14, 8, 0.96),
                   (16, 16, 0.60), (16, 16, 0.96), (16, 16, 1.40),
                   (18, 32, 0.96)):
    nmax = J
    b, phi, cat, rho_c, rho0 = case(J, Kz, rho, nmax, seed=7)
    ex = exact(b, phi, cat, rho_c, rho0, nmax)
    row = f"{J:>3}{Kz:>4}{rho:6.2f}"
    t0 = time.time(); ge, lam = log_Z_gauss(b, phi, cat, rho_c, rho0, nmax, "exact"); t_ge = (time.time()-t0)*1e3
    t0 = time.time(); gm, _   = log_Z_gauss(b, phi, cat, rho_c, rho0, nmax, "mf");    t_gm = (time.time()-t0)*1e3
    t0 = time.time(); q8  = qmc(b, phi, cat, rho_c, rho0, nmax, 8);   t_q8  = (time.time()-t0)*1e3
    t0 = time.time(); q128= qmc(b, phi, cat, rho_c, rho0, nmax, 128); t_q128= (time.time()-t0)*1e3
    print(row + f"{lam:9.3f}{ex:10.4f}{ge-ex:+14.5f}{t_ge:7.1f}{gm-ex:+10.5f}{t_gm:7.1f}"
                f"{q8-ex:+9.4f}{t_q8:7.1f}{q128-ex:+10.5f}{t_q128:7.1f}", flush=True)
