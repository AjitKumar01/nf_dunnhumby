"""Complete-the-basket ranked on pi -- the correct scoring function for this model.

Run:  V3_AFFINITY=1 python3 recommend_pi.py

Originally, CONDITIONED on the rest of the basket.

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
from ragged import RaggedModel, set_quad
D=build(); J,N,C,S=(int(D[k]) for k in ("n_item","n_user","n_cat","n_store"))
F=Features(J,S,712); B=Batcher(D,F,120)
bl=torch.load("../../out/v3_run97_best.pt",map_location="cpu",weights_only=False)
_Kz=int((bl.get("quad") or {}).get("Kz", 4))
m=RaggedModel(J=J,N=N,C=C,K=32,Kz=_Kz,nmax=120,R=23,S=S,Kp=8)
m.load_state_dict(bl["model"],strict=False)
co_=torch.zeros(J,dtype=torch.long)
co_[torch.as_tensor(D["line_item"],dtype=torch.long)]=torch.as_tensor(D["line_cat"],dtype=torch.long)
with torch.no_grad(): m.cat_of.copy_(co_)
# The integrator comes from the CHECKPOINT, not from this file.  It used to read
# smolyak_grid(4, 8) unconditionally, which against a Kz=128 checkpoint is not
# just inaccurate -- the nodes are the wrong shape.
_q=bl.get("quad") or {}
print("[rec] log Z: "+set_quad(m, _q.get("quad_q",8), _q.get("qmc_n",0),
                               _q.get("qmc_seed",0), probe=_q.get("probe", 8), steps=_q.get("steps", 4), chunk=_q.get("chunk", 0)), flush=True)
m.double().eval()
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
    idx=idx[np.random.default_rng(12345).permutation(len(idx))][:2000]
    rng=np.random.default_rng(0)
    R={k:[] for k in ("ours: pi | rest","ours: pi UNconditioned","ours: b only",
                      "popularity","co-purchase","random")}
    for k0 in range(0,len(idx),24):
        ix,ctx,lctx,hh,LI,LT,LC,LU=B.make(idx[k0:k0+24]); m.house,m.ctx=hh,ctx
        with torch.no_grad(): bf=m.b_flat(ix); r0=m.rho_0()
        # ONE backward pass per BATCH, not per trip.  _log_Z_quad computes pi for every
        # trip in the batch at once; calling it inside the per-trip loop threw away 23 of
        # every 24 results, and doing it for both the conditioned and unconditioned scores
        # made that 48x.  Each trip's rest-of-basket is pinned into the same b0, so one
        # pass serves them all -- the same structure fit.py's rec_eval uses.
        hold={}; b0=bf.detach().clone()
        st=rng.bit_generator.state
        for b in range(ix.B):
            basket=LI[LT==b]
            if len(basket)<2: continue
            hid=int(basket[rng.integers(len(basket))]); hold[b]=hid
            sel=(ix.item_trip==b).nonzero().flatten()
            rest=torch.as_tensor([int(x) for x in basket if int(x)!=hid],dtype=torch.long)
            keep=torch.isin(ix.item[sel],rest)
            b0[sel[keep]]=b0[sel[keep]]+FORCE
        rng.bit_generator.state=st                 # replay the same holdouts below
        b0=b0.requires_grad_(True); m._b_override=b0
        lz=m._log_Z_quad(ix,True,False,False,False)
        pig=torch.autograd.grad(lz.sum(),b0)[0].detach(); m._b_override=None
        bu=bf.detach().clone().requires_grad_(True); m._b_override=bu
        lzu=m._log_Z_quad(ix,True,False,False,False)
        piu=torch.autograd.grad(lzu.sum(),bu)[0].detach(); m._b_override=None
        for b in range(ix.B):
            sel=(ix.item_trip==b).nonzero().flatten()
            items=ix.item[sel]; bv=bf[sel]
            basket=LI[LT==b]
            if len(basket)<2: continue
            hid=int(basket[rng.integers(len(basket))])
            assert hid==hold[b], "holdout replay diverged"
            rest=torch.as_tensor([int(x) for x in basket if int(x)!=hid],dtype=torch.long)
            pos=(items==hid).nonzero().flatten()
            if len(rest)==0 or len(pos)==0: continue
            p=int(pos[0])
            inb=torch.zeros(len(items),dtype=torch.bool); inb[torch.isin(items,rest)]=True
            with torch.no_grad():
                nc=torch.bincount(m.cat_of[rest],minlength=C).double()
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
