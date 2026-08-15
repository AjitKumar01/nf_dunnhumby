"""What does the data require of the interaction, and can this construction supply it?

The pairwise term contributes exp(phi_j . phi_k) to a pair's odds of appearing together.
So the question is arithmetic: what LIFT do real baskets show over independence, what
phi_j.phi_k would produce it, and is that compatible with lambda_max < 1 -- the condition
section 14 needs for the normaliser's importance sampler to work at all.
"""
import sys, math, numpy as np, torch
sys.path.insert(0,'../v3'); torch.set_default_dtype(torch.float64)
from data import build; from features import Features; from fit import Batcher
from ragged import RaggedModel

D=build(); lp,li=D["line_ptr"],D["line_item"]
tr=np.flatnonzero(D["trip_split"]==0)[:40000]
J=int(D["n_item"])
cnt=np.zeros(J); pair={}
for t in tr:
    a,b=int(lp[t]),int(lp[t+1]); s=sorted(set(int(x) for x in li[a:b]))
    for j in s: cnt[j]+=1
    for i in range(len(s)):
        for k in range(i+1,len(s)): pair[(s[i],s[k])]=pair.get((s[i],s[k]),0)+1
n=len(tr); p=cnt/n
top=[k for k,v in sorted(pair.items(),key=lambda x:-x[1])[:300]]
lift=np.array([ (pair[k]/n)/max(p[k[0]]*p[k[1]],1e-12) for k in top])
print(f"1. OBSERVED dependence, 300 commonest pairs over {n:,} training trips")
print(f"   lift over independence: median {np.median(lift):8.1f}  "
      f"p25 {np.percentile(lift,25):7.1f}  p75 {np.percentile(lift,75):7.1f}")
need = math.log(np.median(lift))
print(f"   -> the pairwise energy needed is log(lift) = {need:.2f} for a typical common pair")

m=RaggedModel(J=J,N=int(D["n_user"]),C=int(D["n_cat"]),K=32,Kz=12,nmax=120,R=23,
              S=int(D["n_store"]),Kp=8)
m.load_state_dict(torch.load('../../out/v3_run23_cal.pt',map_location='cpu')); m.double().eval()
phi=m.phi.detach()
nrm=phi.norm(dim=1)
ip=torch.tensor([float(phi[a]@phi[b]) for a,b in top])
print(f"\n2. WHAT THE FITTED MODEL SUPPLIES")
print(f"   ||phi|| mean {float(nrm.mean()):.4f}   max {float(nrm.max()):.4f}")
print(f"   phi_j.phi_k on those same pairs: median {float(ip.median()):+.4f}  "
      f"max {float(ip.max()):+.4f}")
print(f"   -> supplied lift = exp({float(ip.median()):.4f}) = {math.exp(float(ip.median())):.3f}x"
      f"   against a required {np.median(lift):.0f}x")

print(f"\n3. IS THE REQUIRED STRENGTH COMPATIBLE WITH STABILITY?")
En=10.0
for tag, e in (("to match the data", need),):
    r = math.sqrt(e)                      # ||phi|| if the pair is aligned
    print(f"   {tag}: phi_j.phi_k = {e:.2f} needs ||phi|| ~ {r:.2f}")
    print(f"   lambda_max ~ ||phi||^2 * E[n] = {r*r:.2f} * {En:.0f} = {r*r*En:.1f}"
          f"   (section 14 needs < 1)")
print(f"   the cap that lambda_max < 1 allows at E[n]={En:.0f}: "
      f"||phi|| <= {math.sqrt(1.0/En):.3f}, i.e. phi.phi <= {1.0/En:.3f}, lift <= "
      f"{math.exp(1.0/En):.3f}x")

print(f"\n4. WHERE THE FITTED phi ACTUALLY SPENDS ITS CAPACITY")
sv=torch.linalg.svdvals(phi); var=(sv**2)/ (sv**2).sum()
print(f"   variance in each direction: " + " ".join(f"{float(v):.3f}" for v in var[:6]))
mu=phi.mean(0); print(f"   ||mean phi|| {float(mu.norm()):.4f}   "
      f"mean ||phi|| {float(nrm.mean()):.4f}   "
      f"ratio {float(mu.norm()/nrm.mean()):.3f}")
proj=(phi@mu)/mu.norm()
print(f"   projection on the mean direction: mean {float(proj.mean()):+.4f} "
      f"sd {float(proj.std()):.4f}")
print(f"   share of pair energy from the mean direction alone: "
      f"{float((proj[[a for a,b in top]]*proj[[b for a,b in top]]).median()/ip.median()):.2f}")
