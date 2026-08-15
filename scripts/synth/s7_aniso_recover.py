"""Does the anisotropic proposal RECOVER a planted phi, at Kz = 12?

s6 showed it estimates log Z well at fixed parameters.  That is not the same test: the
mixture also looked good at K=4 and still drove phi to its cap in a fit.  What matters is
whether the gradient it produces leads back to the truth.  Exact enumeration is the control.
"""
import itertools, numpy as np, torch, math
torch.set_default_dtype(torch.float64)
J, K = 12, 12
mask = torch.zeros(2**J, J)
for i,bits in enumerate(itertools.product([0,1], repeat=J)):
    for j in range(J):
        if bits[j]: mask[i,j]=1.0
nonempty = mask.sum(1) > 0

def E_of(b, PH):
    v = mask @ PH; sq = mask @ (PH**2).sum(1)
    return mask @ b + 0.5*((v*v).sum(1) - sq)

def mode_and_vtop(b, PH, steps=6):
    z = torch.zeros(1, K)
    for _ in range(steps):
        zz = z.detach().requires_grad_(True)
        w = torch.exp(b + zz @ PH.T - 0.5*(PH**2).sum(1))
        z = torch.autograd.grad(torch.log1p(w).sum(), zz)[0]
    zh = z.detach()
    w = torch.exp(b + zh @ PH.T - 0.5*(PH**2).sum(1))[0]
    pi = (w/(1+w)).clamp(1e-12, 1-1e-12)
    L = (PH.detach() * (pi*(1-pi)).unsqueeze(1)).T @ PH.detach()
    ev, V = torch.linalg.eigh(L)
    return zh, V[:, -1]

def logZ_aniso(b, PH, nd, gen, s=2.0):
    zh, vtop = mode_and_vtop(b, PH)
    e0 = torch.randn(nd, K, generator=gen)
    proj = e0 @ vtop
    eps = e0 + (s-1.0)*proj.unsqueeze(1)*vtop.unsqueeze(0)
    r = eps @ vtop
    perp = eps - r.unsqueeze(1)*vtop.unsqueeze(0)
    lq = -0.5*(perp.pow(2).sum(1) + (r/s)**2) - math.log(s)
    zs = zh + eps
    w = torch.exp(b.unsqueeze(0) + zs @ PH.T - 0.5*(PH**2).sum(1).unsqueeze(0))
    lf = torch.expm1(torch.log1p(w).sum(1)).clamp_min(1e-300).log()
    lw = (-0.5*(zs**2).sum(1) + lf) - lq
    return torch.logsumexp(lw,0) - math.log(nd)

def logZ_single(b, PH, nd, gen):
    zh, _ = mode_and_vtop(b, PH)
    eps = torch.randn(nd, K, generator=gen)
    zs = zh + eps
    w = torch.exp(b.unsqueeze(0) + zs @ PH.T - 0.5*(PH**2).sum(1).unsqueeze(0))
    lf = torch.expm1(torch.log1p(w).sum(1)).clamp_min(1e-300).log()
    lw = (-0.5*(zs**2).sum(1) + lf) - (-0.5*eps.pow(2).sum(1))
    return torch.logsumexp(lw,0) - math.log(nd)

print(f"{'true':>6}{'cap':>7}   {'exact':>9}{'single':>9}{'aniso':>9}   "
      f"{'true lift':>10}{'sgl lift':>9}{'ani lift':>9}")
for t in (1.0, 2.0, 3.0):
    v = math.sqrt(t)
    PH_t = torch.zeros(J,K); PH_t[0,0]=v; PH_t[1,0]=v
    b_t = torch.full((J,), -2.0)
    Et = E_of(b_t, PH_t); pt = torch.softmax(Et[nonempty],0)
    idxn = nonempty.nonzero().flatten()
    m0 = mask[idxn]
    pi_t = (m0 * pt.unsqueeze(1)).sum(0)
    joint_t = float((m0[:,0]*m0[:,1]*pt).sum())
    lift_t = joint_t/float(pi_t[0]*pi_t[1])
    rng = np.random.default_rng(0)
    draw = rng.choice(len(idxn), size=40000, p=pt.numpy())
    cnt = torch.zeros(len(idxn))
    for d in draw: cnt[d]+=1
    cap = 1.5*math.sqrt(t)
    res={}
    for mode in ("exact","single","aniso"):
        bh = torch.full((J,), -1.0, requires_grad=True)
        PH = (torch.randn(J,K,generator=torch.Generator().manual_seed(1))*0.1).requires_grad_(True)
        opt = torch.optim.Adam([bh,PH], lr=0.05); gen=torch.Generator().manual_seed(7)
        for step in range(900):
            Ea = E_of(bh,PH)
            if mode=="exact":    lz = torch.logsumexp(Ea[nonempty],0)
            elif mode=="single": lz = logZ_single(bh,PH,256,gen)
            else:                lz = logZ_aniso(bh,PH,256,gen)
            ll = ((Ea[nonempty]-lz)*cnt).sum()/cnt.sum()
            opt.zero_grad(); (-ll).backward(); opt.step()
            with torch.no_grad():
                nn_=PH.norm(dim=1,keepdim=True).clamp_min(1e-12); PH.mul_((cap/nn_).clamp(max=1.0))
        with torch.no_grad():
            Ef = E_of(bh,PH); pf = torch.softmax(Ef[nonempty],0)
            pi_f = (m0*pf.unsqueeze(1)).sum(0)
            jf = float((m0[:,0]*m0[:,1]*pf).sum())
            res[mode]=(float(PH[0]@PH[1]), jf/float(pi_f[0]*pi_f[1]))
    print(f"{t:6.2f}{cap**2:7.2f}   {res['exact'][0]:9.3f}{res['single'][0]:9.3f}"
          f"{res['aniso'][0]:9.3f}   {lift_t:10.3f}{res['single'][1]:9.3f}{res['aniso'][1]:9.3f}",
          flush=True)
