"""Complete-the-basket ranked on pi -- the correct scoring function for this model.

Run:  V3_AFFINITY=1 python3 recommend_pi.py

The earlier comparison ranked on b (and b + phi + rho_c).  b_j is the product's standalone
value, an INPUT, and is uncorrelated with purchase frequency (-0.052); pi_j is what happens
once every product competes under the size law and category term (+0.173).  Bundles built
on pi scored 6.4x better than on b, so the recommendation comparison is worth redoing.

Conditioning: pin the observed rest-of-basket in by raising their b, then take pi over the
whole assortment.  ONE backward pass per trip -- the rest of the basket is fixed, so no
per-candidate re-solve is needed.

Same trips, same held-out items, same baselines as the earlier fair comparison.
"""
import sys; sys.path.insert(0,'/Users/ajit/Projects/Causal/nf_dunnhumby/scripts/v3')
import numpy as np, torch
from collections import defaultdict
torch.set_default_dtype(torch.float64)
from data import build
from features import Features
from fit import Batcher
from ragged import RaggedModel, smolyak_grid
D=build(); J,N,C,S=(int(D[k]) for k in ("n_item","n_user","n_cat","n_store"))
F=Features(J,S,712); B=Batcher(D,F,120)
m=RaggedModel(J=J,N=N,C=C,K=32,Kz=4,nmax=120,R=23,S=S,Kp=8)
bl=torch.load("../../out/v3_run90_best.pt",map_location="cpu",weights_only=False)
m.load_state_dict(bl["model"],strict=False)
co_=torch.zeros(J,dtype=torch.long)
co_[torch.as_tensor(D["line_item"],dtype=torch.long)]=torch.as_tensor(D["line_cat"],dtype=torch.long)
with torch.no_grad(): m.cat_of.copy_(co_)
m.quad=smolyak_grid(4,8); m.double().eval()
lp=D["line_ptr"]; li=D["line_item"]
tr=np.flatnonzero(D["trip_split"]==0); keep=np.zeros(len(li),bool)
for t in tr: keep[int(lp[t]):int(lp[t+1])]=True
pop=np.bincount(li[keep],minlength=J).astype(float); pop/=pop.sum()
cnt_j=np.bincount(li[keep],minlength=J).astype(float)+1.0
cop=defaultdict(float)
for t in tr:
    itms=np.unique(li[int(lp[t]):int(lp[t+1])])
    if not (2<=len(itms)<=40): continue
    for x in itms:
        for y in itms:
            if x!=y: cop[(int(x),int(y))]+=1.0
FORCE=6.0
for SPLIT,nm in ((1,"validation"),(2,"test")):
    idx=np.flatnonzero(D["trip_split"]==SPLIT)
    idx=idx[np.random.default_rng(12345).permutation(len(idx))][:384]
    rng=np.random.default_rng(0)
    R={k:[] for k in ("ours: pi | rest","ours: pi UNconditioned","ours: b only",
                      "popularity","co-purchase","random")}
    for k0 in range(0,len(idx),24):
        ix,ctx,lctx,hh,LI,LT,LC,LU=B.make(idx[k0:k0+24]); m.house,m.ctx=hh,ctx
        with torch.no_grad(): bf=m.b_flat(ix); r0=m.rho_0()
        for b in range(ix.B):
            sel=(ix.item_trip==b).nonzero().flatten()
            items=ix.item[sel]; bv=bf[sel]
            basket=LI[LT==b]
            if len(basket)<2: continue
            hid=int(basket[rng.integers(len(basket))])
            rest=torch.as_tensor([int(x) for x in basket if int(x)!=hid],dtype=torch.long)
            pos=(items==hid).nonzero().flatten()
            if len(rest)==0 or len(pos)==0: continue
            p=int(pos[0])
            inb=torch.zeros(len(items),dtype=torch.bool); inb[torch.isin(items,rest)]=True
            # pi conditioned on the rest of the basket
            b0=bf.detach().clone(); b0[sel[inb]]=b0[sel[inb]]+FORCE
            m._b_override=b0.requires_grad_(True)
            lz=m._log_Z_quad(ix,True,False,False,False)
            pig=torch.autograd.grad(lz.sum(),m._b_override)[0].detach(); m._b_override=None
            # UNCONDITIONED: pi with nothing pinned in.  Still uses lam, taste, price,
            # promo, season, store and the competition structure -- everything except WHICH
            # items are already in the cart.
            bu=bf.detach().clone().requires_grad_(True)
            m._b_override=bu
            lzu=m._log_Z_quad(ix,True,False,False,False)
            piu=torch.autograd.grad(lzu.sum(),bu)[0].detach(); m._b_override=None
            with torch.no_grad():
                pair=m.phi[items]@m.phi[rest].sum(0)
                pair=pair-inb.double()*(m.phi[items]*m.phi[items]).sum(-1)
                nc=torch.bincount(m.cat_of[rest],minlength=C).double()
                rc=-m.rho_c[m.cat_of[items]]*nc[m.cat_of[items]]
                dr=r0[min(len(rest)+1,m.nmax)]-r0[min(len(rest),m.nmax)]
            cs=torch.tensor([sum(cop.get((int(j),int(k)),0.0) for k in rest.tolist())/cnt_j[int(j)]
                             for j in items.tolist()],dtype=torch.float64)
            cand={"ours: pi | rest":pig[sel],"ours: pi UNconditioned":piu[sel],
                  "ours: b only":bv,
                  "popularity":torch.as_tensor(pop[items.numpy()]).clamp_min(1e-12).log(),
                  "co-purchase":cs,
                  "random":torch.as_tensor(np.random.default_rng(b+k0).random(len(items)))}
            for k,sc in cand.items():
                sc=sc.clone(); sc[inb]=-float("inf")
                R[k].append(int((sc>sc[p]).sum())+1)
    print(f"\n{nm}: {len(R['popularity'])} cases, same trips and held-out items")
    print(f"  {'method':>20}{'MRR':>9}{'R@10':>8}{'R@100':>8}{'median':>9}")
    for k,v in R.items():
        r=np.array(v,float)
        print(f"  {k:>20}{np.mean(1/r):9.4f}{100*(r<=10).mean():7.1f}%{100*(r<=100).mean():7.1f}%"
              f"{np.median(r):9.0f}")
