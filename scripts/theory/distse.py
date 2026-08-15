"""The other distributions, per seed, with standard errors -- for both checkpoints.

alldist.py reported point estimates from 8 pooled replications, so the run22-vs-run23
differences (categories 1.61 vs 1.58, spend 1.71 vs 1.70, item rates 0.58 vs 0.59,
co-occurrence 0.08 vs 0.08) had no error bars and could not be called real or noise.
Here each replication is kept separate so every ratio carries one.
"""
import sys, math, numpy as np, torch
sys.path.insert(0,'../v3'); torch.set_default_dtype(torch.float64)
from data import build; from features import Features; from fit import Batcher
from ragged import RaggedModel

D=build(); F=Features(int(D["n_item"]),int(D["n_store"]),712); Bt=Batcher(D,F,120)
J,N,C,S=(int(D[k]) for k in ("n_item","n_user","n_cat","n_store"))
cat_of=np.zeros(J,np.int64); lp,li,lc,lu=D["line_ptr"],D["line_item"],D["line_cat"],D["line_units"]
cat_of[li]=lc
logp=torch.from_numpy(np.load('../../basket_input/log_price.npy').astype(np.float64))
tst=np.flatnonzero(D["trip_split"]==2)[:128]; day=D["trip_day"]
SEEDS=6

o_sz,o_cats,o_spend,o_items,o_pairs=[],[],[],[],{}
for t in tst:
    a,b=int(lp[t]),int(lp[t+1]); d=int(day[t]); it,u=li[a:b],lu[a:b]
    if len(it)==0: continue
    o_sz.append(len(it)); o_cats.append(len(set(cat_of[it].tolist())))
    o_spend.append(float((np.exp(logp[it,d].numpy())*u).sum())); o_items.extend(it.tolist())
    s=sorted(set(int(x) for x in it))
    for i in range(len(s)):
        for j in range(i+1,len(s)): o_pairs[(s[i],s[j])]=o_pairs.get((s[i],s[j]),0)+1
o_sz=np.array(o_sz); cnt_o=np.bincount(np.array(o_items),minlength=J)/len(tst)
top=np.argsort(-cnt_o)[:200]
pairs200=[k for k,v in sorted(o_pairs.items(),key=lambda x:-x[1])[:200]]
po=np.array([o_pairs[k]/len(tst) for k in pairs200])

def stats(m,seed):
    g=torch.Generator().manual_seed(seed)
    sz,cats,spend,items,prs=[],[],[],[],{}
    for k in range(0,len(tst),32):
        sub=tst[k:k+32]; ix,ctx,lctx,hh,*_=Bt.make(sub); m.house,m.ctx=hh,ctx
        for bi,bk in enumerate(m.sample(ix,n_draws=16,generator=g)):
            if not bk: continue
            it=torch.as_tensor(bk,dtype=torch.long); d=int(day[sub[bi]])
            sz.append(len(bk)); cats.append(len(set(cat_of[bk].tolist())))
            spend.append(float(torch.exp(logp[it,d]).sum()*1.34))
            items.extend(bk)
            for i in range(len(bk)):
                for j in range(i+1,len(bk)):
                    prs[(bk[i],bk[j])]=prs.get((bk[i],bk[j]),0)+1
    cs=np.bincount(np.array(items),minlength=J)/len(tst)
    ps=np.array([prs.get(k,0)/len(tst) for k in pairs200])
    return (np.mean(sz),np.percentile(sz,90),np.mean(cats),np.mean(spend),
            cs[top].mean()/cnt_o[top].mean(), ps.mean()/po.mean(), (ps==0).sum())

def se(x): x=np.asarray(x,float); return x.mean(), x.std(ddof=1)/math.sqrt(len(x))
print(f"{len(tst)} test trips, {SEEDS} seeds")
print(f"observed: mean size {o_sz.mean():.2f}  p90 {np.percentile(o_sz,90):.1f}  "
      f"cats {np.mean(o_cats):.2f}  spend {np.mean(o_spend):.2f}")
for tag,ck in (("run23+recal",'v3_run23_cal.pt'),("run24+recal",'v3_run24_cal.pt')):
    m=RaggedModel(J=J,N=N,C=C,K=32,Kz=12,nmax=120,R=23,S=S,Kp=8)
    m.load_state_dict(torch.load(f'../../out/{ck}',map_location='cpu')); m.double().eval()
    rows=np.array([stats(m,300+s) for s in range(SEEDS)])
    nm=["mean size","p90 size","categories","spend","item-rate ratio",
        "pair-cooc ratio","pairs missing/200"]
    ref=[o_sz.mean(),np.percentile(o_sz,90),np.mean(o_cats),np.mean(o_spend),1.0,1.0,0]
    print(f"\n{tag}")
    for i,n_ in enumerate(nm):
        mu,s_=se(rows[:,i])
        r=f"   ratio {mu/ref[i]:.2f}" if ref[i] not in (0,1.0) else ""
        print(f"   {n_:20s} {mu:8.3f} +/- {s_:6.3f}{r}")
