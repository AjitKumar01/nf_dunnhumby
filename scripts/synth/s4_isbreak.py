"""At what interaction strength does importance sampling stop recovering the truth?

s3 recovered a planted phi almost exactly by enumerating all 2^J subsets.  The real fit
cannot do that: it estimates log Z by importance sampling over the Gaussian latent.  Same
planted data, same catalogue, two normalisers -- exact vs sampled -- swept over interaction
strength.  Where the sampled fit stops tracking the exact one is the boundary that matters,
and whether it sits below what grocery data needs (phi.phi ~ 2) is the whole question.
"""
import itertools, numpy as np, torch, math
torch.set_default_dtype(torch.float64)
exec(open('s1_represent.py').read().split('J, K, C = ')[0].split('"""',2)[2])

J, K, C = 12, 4, 3
cat = np.array([0,0,0,0, 1,1,1,1, 2,2,2,2])
rho_c0 = torch.zeros(C); rho00 = torch.zeros(J+1)
subs_all = [[j for j in range(J) if bits[j]] for bits in itertools.product([0,1], repeat=J)]
mask = torch.zeros(len(subs_all), J)
for i,S in enumerate(subs_all):
    for j in S: mask[i,j] = 1.0
nonempty = mask.sum(1) > 0

def logZ_exact(b, PH):
    v = mask @ PH; sq = mask @ (PH**2).sum(1)
    E = mask @ b + 0.5*((v*v).sum(1) - sq)
    return torch.logsumexp(E[nonempty], 0), E

def logZ_mix(b, PH, n_draws, gen, scales=(1.0, 2.0)):
    """Defensive mixture proposal: half tight, half wide, scored under the mixture."""
    z = torch.zeros(1, K)
    for _ in range(3):
        zz = z.detach().requires_grad_(True)
        w = torch.exp(b + zz @ PH.T - 0.5*(PH**2).sum(1))
        z = torch.autograd.grad(torch.log1p(w).sum(), zz)[0]
    zh = z.detach()
    per = n_draws // len(scales)
    eps = torch.cat([torch.randn(per, K, generator=gen)*sc for sc in scales])
    comp = torch.stack([(-0.5*(eps/sc).pow(2).sum(1) - K*math.log(sc)) for sc in scales])
    lq = torch.logsumexp(comp, 0) - math.log(len(scales))
    zs = zh + eps
    w = torch.exp(b.unsqueeze(0) + zs @ PH.T - 0.5*(PH**2).sum(1).unsqueeze(0))
    lf = torch.expm1(torch.log1p(w).sum(1)).clamp_min(1e-300).log()
    lw = (-0.5*(zs**2).sum(1) + lf) - lq
    return torch.logsumexp(lw, 0) - math.log(len(eps))


def logZ_is(b, PH, n_draws, gen):
    """Theorem 1 by importance sampling at the mode -- the real fit's estimator."""
    z = torch.zeros(1, K)
    for _ in range(3):
        zz = z.detach().requires_grad_(True)
        w = torch.exp(b + zz @ PH.T - 0.5*(PH**2).sum(1))
        # log f(z) = sum_j log(1 + w_j), NOT log(1 + sum_j w_j).  With the wrong form the
        # mode lands in the wrong place, the proposal is misplaced, log Z is understated and
        # the fit is rewarded for growing phi without bound -- it ran to phi.phi = 34,000.
        lf = torch.log1p(w).sum()
        z = torch.autograd.grad(lf, zz)[0]
    zh = z.detach()
    noise = torch.randn(n_draws, K, generator=gen)
    zs = zh + noise
    w = torch.exp(b.unsqueeze(0) + zs @ PH.T - 0.5*(PH**2).sum(1).unsqueeze(0))
    f_minus_1 = torch.expm1(torch.log1p(w).sum(1))      # prod(1+w) - 1, drops n=0
    lp = -0.5*(zs**2).sum(1) + torch.log(f_minus_1.clamp_min(1e-300))
    lq = -0.5*(noise**2).sum(1)
    lw = lp - lq
    return torch.logsumexp(lw, 0) - math.log(n_draws)

import sys
ND = int(sys.argv[1]) if len(sys.argv)>1 else 32
print(f"draws = {ND}")
print(f"{'phi.phi':>8}{'lam_max':>9}{'exact':>11}{'IS':>9}{'MIX':>9}{'true lift':>12}{'IS lift':>9}{'MIX lift':>9}")
for t in (0.5, 1.0, 2.0, 3.0, 4.0):
    v = math.sqrt(t)
    PHI_t = torch.zeros(J, K); PHI_t[0,0]=v; PHI_t[1,0]=v
    b_t = torch.full((J,), -2.0)
    subs, p = enumerate_model(b_t, PHI_t, rho_c0, cat, rho00)
    _, lift_t = marginals_and_lift(subs, p, J, [(0,1)])
    rng = np.random.default_rng(0)
    idx = rng.choice(len(subs), size=40000, p=p)
    cnt = torch.zeros(len(subs))
    for i in idx: cnt[i] += 1
    cap = 1.5*math.sqrt(t)          # generous: 1.5x what the truth needs
    res = {}
    for mode in ("exact","is","mix"):
        bh = torch.full((J,), -1.0, requires_grad=True)
        PH = (torch.randn(J,K,generator=torch.Generator().manual_seed(1))*0.1).requires_grad_(True)
        opt = torch.optim.Adam([bh,PH], lr=0.05)
        gen = torch.Generator().manual_seed(7)
        for step in range(1200):
            vv = mask @ PH; sq = mask @ (PH**2).sum(1)
            E = mask @ bh + 0.5*((vv*vv).sum(1) - sq)
            if mode == "exact":   lz = logZ_exact(bh,PH)[0]
            elif mode == "is":    lz = logZ_is(bh,PH,ND,gen)
            else:                 lz = logZ_mix(bh,PH,ND,gen)
            ll = ((E - lz)*cnt)[nonempty].sum()/cnt[nonempty].sum()
            opt.zero_grad(); (-ll).backward(); opt.step()
            with torch.no_grad():      # the real fit's phi cap; without it an understated
                nn_ = PH.norm(dim=1, keepdim=True).clamp_min(1e-12)   # log Z rewards phi
                PH.mul_((cap/nn_).clamp(max=1.0))                     # without bound
        with torch.no_grad():
            s2,p2 = enumerate_model(bh.detach(), PH.detach(), rho_c0, cat, rho00)
            _, lf = marginals_and_lift(s2,p2,J,[(0,1)])
            res[mode] = (float(PH[0]@PH[1]), lf[(0,1)])
    pi_t,_ = marginals_and_lift(subs,p,J,[(0,1)])
    lam = t*float(np.sum(pi_t*(1-pi_t)))
    print(f"{t:8.2f}{lam:9.2f}{res['exact'][0]:11.3f}{res['is'][0]:9.3f}{res['mix'][0]:9.3f}"
          f"{lift_t[(0,1)]:12.3f}{res['is'][1]:9.3f}{res['mix'][1]:9.3f}", flush=True)
