"""Which proposal gets log Z right at the strength grocery data needs?

s4 showed importance sampling does not merely add noise: it UNDERSTATES log Z, which makes
the likelihood reward larger phi until a cap stops it.  Fixing the fit therefore means
fixing the proposal.  Here log Z is computed exactly by enumeration at the PLANTED
parameters and compared against candidates, which is far faster than fitting each one.

f(z) = prod_j (1 + w_j),  w_j = exp(b_j + z.phi_j - |phi_j|^2/2), so log f is convex in z
and the integrand N(z;0,I) f(z) is heavier-tailed than the unit Gaussian the current
proposal uses.  The candidates widen or fatten it.
"""
import itertools, numpy as np, torch, math
torch.set_default_dtype(torch.float64)
J, K = 12, 4
mask = torch.zeros(2**J, J)
for i,bits in enumerate(itertools.product([0,1], repeat=J)):
    for j in range(J):
        if bits[j]: mask[i,j]=1.0
nonempty = mask.sum(1) > 0

def logZ_exact(b, PH):
    v = mask @ PH; sq = mask @ (PH**2).sum(1)
    E = mask @ b + 0.5*((v*v).sum(1) - sq)
    return float(torch.logsumexp(E[nonempty], 0))

def mode(b, PH, steps=8):
    z = torch.zeros(1, K)
    for _ in range(steps):
        zz = z.detach().requires_grad_(True)
        w = torch.exp(b + zz @ PH.T - 0.5*(PH**2).sum(1))
        z = torch.autograd.grad(torch.log1p(w).sum(), zz)[0]
    return z.detach()

def logZ_mix(b, PH, nd, gen, scales=(1.0, 2.0)):
    """Defensive mixture: half the draws tight, half wide, weighted by the MIXTURE density.

    A single scale trades low-strength accuracy against high-strength bias -- N(mode,I) is
    exact at phi.phi = 0.5 and off by -0.45 at 4.0, while N(mode,2I) is the reverse.  Drawing
    from a mixture and scoring under the mixture density keeps the estimator unbiased at both
    ends without having to know which regime it is in.
    """
    zh = mode(b, PH)
    per = nd // len(scales)
    eps = torch.cat([torch.randn(per, K, generator=gen) * s for s in scales])
    # mixture log-density, equal weights
    comp = torch.stack([(-0.5*(eps/s).pow(2).sum(1) - K*math.log(s)) for s in scales])
    lq = torch.logsumexp(comp, 0) - math.log(len(scales))
    zs = zh + eps
    w = torch.exp(b.unsqueeze(0) + zs @ PH.T - 0.5*(PH**2).sum(1).unsqueeze(0))
    lf = torch.expm1(torch.log1p(w).sum(1)).clamp_min(1e-300).log()
    lw = (-0.5*(zs**2).sum(1) + lf) - lq
    ess = float(1.0/torch.softmax(lw,0).pow(2).sum()/len(eps))
    return float(torch.logsumexp(lw,0) - math.log(len(eps))), ess


def logZ_is(b, PH, nd, gen, kind="gauss", scale=1.0, df=3):
    zh = mode(b, PH)
    if kind == "gauss":
        eps = torch.randn(nd, K, generator=gen) * scale
        lq = -0.5*(eps/scale).pow(2).sum(1) - K*math.log(scale)
    elif kind == "t":                                   # multivariate t, heavier tails
        g_ = torch.randn(nd, K, generator=gen)
        u = torch.distributions.Chi2(df).sample((nd,))
        eps = g_ * torch.sqrt(df/u).unsqueeze(1) * scale
        r2 = (eps/scale).pow(2).sum(1)
        lq = -0.5*(df+K)*torch.log1p(r2/df) - K*math.log(scale)
    zs = zh + eps
    w = torch.exp(b.unsqueeze(0) + zs @ PH.T - 0.5*(PH**2).sum(1).unsqueeze(0))
    lf = torch.expm1(torch.log1p(w).sum(1)).clamp_min(1e-300).log()
    lw = (-0.5*(zs**2).sum(1) + lf) - lq
    ess = float(1.0/torch.softmax(lw,0).pow(2).sum()/nd)
    return float(torch.logsumexp(lw,0) - math.log(nd)), ess

SEEDS = 12
cands = [("N(mode,I)","gauss",1.0), ("N(mode,1.5I)","gauss",1.5),
         ("mix(1,2)","mix",0), ("mix(1,2,4)","mix3",0)]
print(f"bias (mean est - exact) and RMSE over {SEEDS} seeds, 512 draws\n")
print(f"{'phi.phi':>8}{'exact':>9}  " + "".join(f"{n:>22}" for n,_,_ in cands))
print(f"{'':17}  " + "".join(f"{'bias':>10}{'rmse':>7}{'ESS':>5}" for _ in cands))
for t in (0.5, 1.0, 2.0, 3.0, 4.0):
    v = math.sqrt(t)
    PH = torch.zeros(J,K); PH[0,0]=v; PH[1,0]=v
    b = torch.full((J,), -2.0)
    ex = logZ_exact(b, PH)
    row = f"{t:8.2f}{ex:9.4f}  "
    for name, kind, sc in cands:
        es, ee = [], []
        for sd in range(SEEDS):
            g = torch.Generator().manual_seed(sd)
            if kind == "mix":
                e_, s_ = logZ_mix(b, PH, 512, g, scales=(1.0, 2.0))
            elif kind == "mix3":
                e_, s_ = logZ_mix(b, PH, 512, g, scales=(1.0, 2.0, 4.0))
            else:
                e_, s_ = logZ_is(b, PH, 512, g, kind=kind, scale=sc)
            es.append(e_); ee.append(s_)
        es = np.array(es)
        row += f"{es.mean()-ex:+10.4f}{np.sqrt(((es-ex)**2).mean()):7.4f}{np.mean(ee):5.2f}"
    print(row, flush=True)
