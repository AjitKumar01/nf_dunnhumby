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
    from ragged import set_quad
    _q=blob.get("quad") or {} if isinstance(blob,dict) else {}
    log("log Z: "+set_quad(m, _q.get("quad_q",a.quad_q), _q.get("qmc_n",0),
                           _q.get("qmc_seed",0), Kz=a.Kz, probe=_q.get("probe", 8),
                           steps=_q.get("steps", 4), chunk=_q.get("chunk", 0),
                           qmc_reps=_q.get("reps", 1),
                           size_bands=_q.get("size_bands", 0),
                           size_steps=_q.get("size_steps", 2),
                           mode_logtol=_q.get("mode_logtol", 8.0),
                           mode_sep=_q.get("mode_sep", 1.0),
                           mix_n=_q.get("mix_n", 0)))
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
            # LOCKSTEP over trips.  _log_Z_quad computes pi for every trip in the batch
            # at once, so running the greedy inside the per-trip loop threw away 23 of
            # every 24 results at each of k steps -- 24*k full passes per batch instead of
            # k.  Different trips are independent, so all of them can advance one step
            # together: pin each trip's current selection, take ONE backward pass, and let
            # each trip pick its own argmax from its own slots.
            slots={b:(ix.item_trip==b).nonzero().flatten() for b in range(ix.B)}
            truth={b:set(int(x) for x in LI[LT==b]) for b in range(ix.B)}
            cur={b:[] for b in range(ix.B)}
            taken={b:torch.zeros(len(slots[b]),dtype=torch.bool) for b in range(ix.B)}
            b0=bf.detach().clone()
            for step in range(a.k):
                bb=b0.clone().requires_grad_(True)
                m._b_override=bb
                lz=m.log_Z(ix,drop_empty=True)
                pig=torch.autograd.grad(lz.sum(),bb)[0].detach()
                m._b_override=None
                for b in range(ix.B):
                    sc=pig[slots[b]].clone()
                    sc[taken[b]]=-float("inf")
                    j=int(sc.argmax()); taken[b][j]=True
                    cur[b].append(int(ix.item[slots[b][j]]))
                    b0[int(slots[b][j])]=b0[int(slots[b][j])]+a.force
            for b in range(ix.B):
                items=ix.item[slots[b]]
                with torch.no_grad():
                    tv=(m.theta_c()[hh[b]]*m.alpha[items]).sum(-1)
                bp=[int(items[i]) for i in torch.topk(torch.as_tensor(pop[items.numpy()]),a.k).indices]
                bt=[int(items[i]) for i in torch.topk(tv,a.k).indices]
                for nm,bun in (("model",cur[b]),("popularity",bp),("taste (theta.alpha)",bt)):
                    hits[nm].append(len(set(bun)&truth[b])/a.k)
                    ncat[nm].append(len(set(int(m.cat_of[x]) for x in bun)))
                if shown<a.show and b==0:
                    shown+=1
                    log(f"  household {int(hh[b])}, bundle of {a.k}:")
                    for x in cur[b]:
                        mark="*" if x in truth[b] else " "
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
