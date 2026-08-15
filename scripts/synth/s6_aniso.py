"""Widen the proposal only along Lambda's large eigendirections.

The isotropic mixture failed at Kz = 12: widening every dimension by 2x multiplies the
sampled volume by 2^12 ~ 4,000, so the wide component occasionally lands where log f is
enormous, one weight dominates, and log Z drifts UPWARD with more draws (12.03 -> 12.30 ->
12.47, sd 0.015 -> 0.47) while the single proposal converged to sd 0.0008.

But the target is only broad where Lambda = Cov(sum_j phi_j x_j | z) is large, and Lambda is
low rank -- effective rank 1-2 in every fitted model.  Widening along its top eigenvector
alone costs a factor of s in volume rather than s^K.

Proposal: N(mode, I + (s^2 - 1) v v') with v the top eigenvector of Lambda.  Tested at
K = 4 AND K = 12 so the dimension dependence that killed the mixture is visible here.
"""
import itertools, numpy as np, torch, math
torch.set_default_dtype(torch.float64)

def build(J, K, t, seed=0):
    v = math.sqrt(t)
    PH = torch.zeros(J, K); PH[0,0]=v; PH[1,0]=v
    g = torch.Generator().manual_seed(seed)
    PH[2:] = torch.randn(J-2, K, generator=g) * 0.15      # background, not just the pair
    return torch.full((J,), -2.0), PH

def logZ_exact(b, PH, mask, nonempty):
    v = mask @ PH; sq = mask @ (PH**2).sum(1)
    E = mask @ b + 0.5*((v*v).sum(1) - sq)
    return float(torch.logsumexp(E[nonempty], 0))

def mode_and_lambda(b, PH, K, steps=10):
    z = torch.zeros(1, K)
    for _ in range(steps):
        zz = z.detach().requires_grad_(True)
        w = torch.exp(b + zz @ PH.T - 0.5*(PH**2).sum(1))
        z = torch.autograd.grad(torch.log1p(w).sum(), zz)[0]
    zh = z.detach()
    w = torch.exp(b + zh @ PH.T - 0.5*(PH**2).sum(1))[0]
    pi = (w/(1+w)).clamp(1e-12, 1-1e-12)
    L = (PH * (pi*(1-pi)).unsqueeze(1)).T @ PH            # [K,K]
    ev, V = torch.linalg.eigh(L)
    return zh, float(ev[-1]), V[:, -1]

def logZ(b, PH, K, nd, gen, kind="single", s=2.0):
    zh, lam, vtop = mode_and_lambda(b, PH, K)
    if kind == "single":
        eps = torch.randn(nd, K, generator=gen)
        lq = -0.5*eps.pow(2).sum(1)
    elif kind == "iso":                                   # the mixture that failed
        per = nd//2
        eps = torch.cat([torch.randn(per,K,generator=gen),
                         torch.randn(nd-per,K,generator=gen)*s])
        comp = torch.stack([-0.5*eps.pow(2).sum(1),
                            -0.5*(eps/s).pow(2).sum(1) - K*math.log(s)])
        lq = torch.logsumexp(comp,0) - math.log(2.0)
    elif kind == "aniso":                                 # widen along v_top only
        e0 = torch.randn(nd, K, generator=gen)
        proj = e0 @ vtop
        eps = e0 + (s - 1.0) * proj.unsqueeze(1) * vtop.unsqueeze(0)
        # density of that linear map: scales only the v_top component
        r = eps @ vtop
        perp = eps - r.unsqueeze(1)*vtop.unsqueeze(0)
        lq = -0.5*(perp.pow(2).sum(1) + (r/s)**2) - math.log(s)
    zs = zh + eps
    w = torch.exp(b.unsqueeze(0) + zs @ PH.T - 0.5*(PH**2).sum(1).unsqueeze(0))
    lf = torch.expm1(torch.log1p(w).sum(1)).clamp_min(1e-300).log()
    lw = (-0.5*(zs**2).sum(1) + lf) - lq
    return float(torch.logsumexp(lw,0) - math.log(nd)), lam

for K in (4, 12):
    J = 12
    mask = torch.zeros(2**J, J)
    for i,bits in enumerate(itertools.product([0,1], repeat=J)):
        for j in range(J):
            if bits[j]: mask[i,j]=1.0
    nonempty = mask.sum(1) > 0
    print(f"\n=== Kz = {K} ===")
    print(f"{'phi.phi':>8}{'lam':>7}{'exact':>10}   " +
          "".join(f"{n:>20}" for n in ("single","iso mix(1,2)","aniso(vtop,2)")))
    for t in (0.5, 2.0, 4.0):
        b, PH = build(J, K, t)
        ex = logZ_exact(b, PH, mask, nonempty)
        row=f"{t:8.2f}"
        lam_shown=None
        cells=[]
        for kind in ("single","iso","aniso"):
            es=[]
            for sd in range(8):
                g=torch.Generator().manual_seed(sd)
                e_,lam = logZ(b,PH,K,512,g,kind=kind)
                es.append(e_)
            lam_shown=lam
            es=np.array(es)
            cells.append(f"{es.mean()-ex:+9.4f}±{es.std(ddof=1):.3f}")
        print(f"{row}{lam_shown:7.2f}{ex:10.4f}   " + "".join(f"{c:>20}" for c in cells), flush=True)
