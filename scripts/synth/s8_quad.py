"""Quadrature along the dominant direction, sampling in the rest.

Importance sampling degrades at phi.phi ~ 3 because the integrand N(z;0,I) f(z) develops a
heavy tail, and the tail lies along ONE direction: Lambda is low rank (effective rank 1-2 in
every fitted model), so f varies sharply along its top eigenvector and is nearly flat in the
other Kz-1 dimensions.

Sampling is the wrong tool for a sharp 1-D integral and the right tool for a flat (Kz-1)-D
one.  Split them:  z = r*v + z_perp, integrate r by Gauss-Hermite (deterministic, no
variance, exact for smooth integrands) and z_perp by a few draws where the integrand barely
moves.  This is Rao-Blackwellisation along the direction that breaks the estimator.

Cost is n_r * n_perp evaluations -- 21 x 8 = 168 against 512 for plain IS.
"""
import itertools, numpy as np, torch, math
torch.set_default_dtype(torch.float64)
J, K = 12, 12
mask = torch.zeros(2**J, J)
for i,bits in enumerate(itertools.product([0,1], repeat=J)):
    for j in range(J):
        if bits[j]: mask[i,j]=1.0
nonempty = mask.sum(1) > 0

def logZ_exact(b, PH):
    v = mask @ PH; sq = mask @ (PH**2).sum(1)
    E = mask @ b + 0.5*((v*v).sum(1) - sq)
    return float(torch.logsumexp(E[nonempty], 0))

def mode_and_vtop(b, PH, steps=10):
    z = torch.zeros(1, K)
    for _ in range(steps):
        zz = z.detach().requires_grad_(True)
        w = torch.exp(b + zz @ PH.T - 0.5*(PH**2).sum(1))
        z = torch.autograd.grad(torch.log1p(w).sum(), zz)[0]
    zh = z.detach()
    w = torch.exp(b + zh @ PH.T - 0.5*(PH**2).sum(1))[0]
    pi = (w/(1+w)).clamp(1e-12,1-1e-12)
    L = (PH * (pi*(1-pi)).unsqueeze(1)).T @ PH
    ev, V = torch.linalg.eigh(L)
    return zh, V[:,-1]

def log_f(b, PH, zs):
    w = torch.exp(b.unsqueeze(0) + zs @ PH.T - 0.5*(PH**2).sum(1).unsqueeze(0))
    return torch.expm1(torch.log1p(w).sum(1)).clamp_min(1e-300).log()

def logZ_is(b, PH, nd, gen):
    zh,_ = mode_and_vtop(b, PH)
    eps = torch.randn(nd, K, generator=gen)
    zs = zh + eps
    lw = (-0.5*(zs**2).sum(1) + log_f(b,PH,zs)) - (-0.5*eps.pow(2).sum(1))
    return float(torch.logsumexp(lw,0) - math.log(nd))

def logZ_quad(b, PH, n_r, n_perp, gen):
    """Gauss-Hermite along v_top, Monte Carlo in the orthogonal complement."""
    zh, v = mode_and_vtop(b, PH)
    xr, wr = np.polynomial.hermite_e.hermegauss(n_r)          # weight exp(-x^2/2)
    xr = torch.tensor(xr); wr = torch.tensor(wr) / math.sqrt(2*math.pi)
    eps = torch.randn(n_perp, K, generator=gen)
    eps = eps - (eps @ v).unsqueeze(1) * v.unsqueeze(0)       # project out v
    # z = r v + eps ; the N(0,I) weight factorises as N(r;0,1) x N(eps;0,I_perp)
    parts = []
    for i in range(n_r):
        zs = xr[i]*v.unsqueeze(0) + eps
        parts.append(torch.log(wr[i].clamp_min(1e-300)) + log_f(b,PH,zs))
    stack = torch.stack(parts)                                 # [n_r, n_perp]
    return float(torch.logsumexp(stack.reshape(-1),0) - math.log(n_perp))

CFG = [(21,8),(11,32),(9,64),(7,128)]
hdr = "".join(f"{f'q({r}x{p_})':>20}" for r,p_ in CFG)
print(f"{'phi.phi':>8}{'exact':>9}{'IS(512)':>20}" + hdr)
for t in (1.0, 2.0, 3.0, 4.0, 6.0):
    v = math.sqrt(t)
    PH = torch.zeros(J,K); PH[0,0]=v; PH[1,0]=v
    g0 = torch.Generator().manual_seed(0)
    PH[2:] = torch.randn(J-2, K, generator=g0)*0.15
    b = torch.full((J,), -2.0)
    ex = logZ_exact(b,PH)
    a=[]
    for sd in range(8):
        g=torch.Generator().manual_seed(sd); a.append(logZ_is(b,PH,512,g))
    a=np.array(a)
    row=f"{t:8.2f}{ex:9.4f}{a.mean()-ex:+11.4f}±{a.std(ddof=1):.3f}"
    for nr,npp in CFG:
        c=[]
        for sd in range(8):
            g=torch.Generator().manual_seed(sd); c.append(logZ_quad(b,PH,nr,npp,g))
        c=np.array(c)
        rmse=np.sqrt(((c-ex)**2).mean())
        row += f"{c.mean()-ex:+11.4f}±{c.std(ddof=1):.2f}"
    print(row, flush=True)
