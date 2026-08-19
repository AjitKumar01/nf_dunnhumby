"""PERSONALISED BUNDLES: build a set, greedily, under the model's own energy.

Adding j to a partial bundle S changes the energy by

    Delta_j(S) = b_jht + sum_{k in S} phi_j'phi_k - rho_c n_c(S) - [rho_0(n+1) - rho_0(n)]

and log Z cancels, so a bundle can be built with no normaliser and no sampling: start empty,
add argmax Delta, repeat.  Every term is doing something different --
  b_jht      personalisation (theta_h.alpha_j), price, promotion, season, store
  phi'phi    complementarity, on the 20 products that carry it
  rho_c      within-category, FITTED NEGATIVE, so it pulls toward the same category
  rho_0      the size law, constant across candidates at a given size

Judged against what the household actually bought on a held-out trip, and against two
baselines that use the same information more crudely.

Run:  V3_AFFINITY=1 python3 bundle.py --ckpt v3_run90_best.pt
"""
import argparse, os
import numpy as np, pandas as pd, torch
torch.set_default_dtype(torch.float64)
from data import build
from features import Features
from fit import Batcher
from ragged import RaggedModel

def log(m): print(f"[bun] {m}", flush=True)

def main(a):
    D=build(); J,N,C,S=(int(D[k]) for k in ("n_item","n_user","n_cat","n_store"))
    F=Features(J,S,712); Bt=Batcher(D,F,a.nmax)
    m=RaggedModel(J=J,N=N,C=C,K=32,Kz=a.Kz,nmax=a.nmax,R=a.R,S=S,Kp=8)
    blob=torch.load(os.path.join("../../out",a.ckpt),map_location="cpu",weights_only=False)
    sd=blob["model"] if isinstance(blob,dict) and blob.get("format")==2 else blob
    miss,_=m.load_state_dict(sd,strict=False)
    assert not [k for k in miss if k not in ("cat_of","price_kappa")], miss
    co=torch.zeros(J,dtype=torch.long)
    co[torch.as_tensor(D["line_item"],dtype=torch.long)]=torch.as_tensor(D["line_cat"],dtype=torch.long)
    with torch.no_grad(): m.cat_of.copy_(co)
    from ragged import smolyak_grid
    m.quad=smolyak_grid(a.Kz,a.quad_q)
    m.double().eval()
    it=pd.read_parquet("../../basket_input/items.parquet").set_index("item_id")
    lp=D["line_ptr"]; li=D["line_item"]
    tr=np.flatnonzero(D["trip_split"]==0); keep=np.zeros(len(li),bool)
    for t in tr: keep[int(lp[t]):int(lp[t+1])]=True
    pop=np.bincount(li[keep],minlength=J).astype(float); pop/=pop.sum()
    va=np.flatnonzero(D["trip_split"]==(1 if a.split=="valid" else 2))
    va=np.array([t for t in va if a.k<=int(lp[t+1])-int(lp[t])<=a.nmax])
    va=np.sort(np.random.default_rng(0).choice(va,size=min(a.n_trips,len(va)),replace=False))
    log(f"{a.ckpt} iter {blob.get('iter','?')}; bundles of {a.k} for {len(va)} held-out trips\n")
    hits={k:[] for k in ("model","popularity","taste (theta.alpha)")}
    ncat={k:[] for k in hits}; shown=0
    r0=m.rho_0().detach()
    if True:
        for k0 in range(0,len(va),a.chunk):
            ix,ctx,lctx,hh,LI,LT,LC,LU=Bt.make(va[k0:k0+a.chunk]); m.house,m.ctx=hh,ctx
            with torch.no_grad(): bf=m.b_flat(ix)
            for b in range(ix.B):
                sel=(ix.item_trip==b).nonzero().flatten()
                items=ix.item[sel]; bv=bf[sel]
                truth=set(int(x) for x in LI[LT==b])
                # --- model: greedy on pi, CONDITIONED on what is already in the bundle ---
                #
                # Ranking on b was wrong.  b_j is the product's standalone value, an INPUT;
                # pi_j = d log(Z-1) / d b_j is what actually happens once 5,455 products
                # compete for ~8 slots under the size law and the category term.  Measured
                # on run90_best, corr(mean b, log purchases) = -0.052 while
                # corr(mean pi, log purchases) = +0.173, so a greedy on b populates the
                # bundle with products nobody buys.
                #
                # Conditioning: forcing the chosen items into the basket by raising their b
                # makes the model re-solve for everything else, so the next pick is the one
                # most likely GIVEN the bundle so far -- complements through phi, category
                # crowding through rho_c, and the size law all act automatically.
                cur=[]; taken=torch.zeros(len(items),dtype=torch.bool)
                b0=bf.detach().clone()
                for step in range(a.k):
                    m._b_override=b0.clone().requires_grad_(True)
                    lz=m._log_Z_quad(ix,True,False,False,False)
                    pig=torch.autograd.grad(lz.sum(),m._b_override)[0].detach()
                    m._b_override=None
                    sc=pig[sel].clone(); sc[taken]=-float("inf")
                    j=int(sc.argmax()); taken[j]=True
                    cur.append(int(items[j]))
                    b0[int(sel[j])]=b0[int(sel[j])]+a.force   # pin it into the basket
                # --- baselines on the same assortment ---
                bp=[int(items[i]) for i in torch.topk(torch.as_tensor(pop[items.numpy()]),a.k).indices]
                with torch.no_grad(): tv=(m.theta_c()[hh[b]]*m.alpha[items]).sum(-1)
                bt=[int(items[i]) for i in torch.topk(tv,a.k).indices]
                for nm,bun in (("model",cur),("popularity",bp),("taste (theta.alpha)",bt)):
                    hits[nm].append(len(set(bun)&truth)/a.k)
                    ncat[nm].append(len(set(int(m.cat_of[x]) for x in bun)))
                if shown<a.show and b==0:
                    shown+=1
                    log(f"  household {int(hh[b])}, bundle of {a.k}:")
                    for x in cur:
                        mark="*" if x in truth else " "
                        log(f"    {mark} {str(it.loc[x,'SUB_COMMODITY_DESC'])[:38]:<38} "
                            f"({str(it.loc[x,'DEPARTMENT'])[:14]})")
    log("")
    log(f"  {'method':>22}{'hit rate':>11}{'distinct categories':>21}")
    for nm in hits:
        log(f"  {nm:>22}{np.mean(hits[nm]):11.3f}{np.mean(ncat[nm]):21.2f}  of {a.k}")
    log(f"\n  hit rate = fraction of the bundle the household actually bought that trip.")
    log(f"  distinct categories shows whether the bundle is varied or collapses into one.")

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--ckpt",default="v3_run90_best.pt"); p.add_argument("--k",type=int,default=5)
    p.add_argument("--n-trips",type=int,default=480)
    p.add_argument("--split",default="valid"); p.add_argument("--chunk",type=int,default=24)
    p.add_argument("--show",type=int,default=3)
    p.add_argument("--no-rhoc",type=int,default=0)
    p.add_argument("--force",type=float,default=6.0,help="b boost that pins a chosen item in")
    p.add_argument("--quad-q",type=int,default=8)
    p.add_argument("--Kz",type=int,default=4); p.add_argument("--nmax",type=int,default=120)
    p.add_argument("--R",type=int,default=23)
    main(p.parse_args())
