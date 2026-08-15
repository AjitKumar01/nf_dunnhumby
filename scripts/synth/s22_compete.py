"""Does phi get spent on basket SIZE instead of on PAIRS -- and does centring stop it?

s21 found rho_0 and phi are redundant for one job: restricting rho_0 to be linear in n cost
almost nothing (KL 0.0022 against 0.0001 free), because phi's pair term is itself quadratic in
n and simply absorbed the role.  So the model has two ways to shape the size law, and only one
way to express complementarity.

That predicts a specific failure.  Matching the size law is easy -- one direction of phi,
loading on every product, does it.  Matching pair structure is hard -- it needs many
directions, each on a few products.  If the likelihood prefers the easy job, phi collapses to
a global size direction and pair lift never appears.  Every real run collapsed to erank 1-2
while co-occurrence stayed at 0.065 against the 1.000 needed.

The truth here has NO global direction: six disjoint pairs, each at phi'phi = 0.92 (the
strength grocery needs) in its own orthogonal direction, plus a rho_0 giving an overdispersed
size law.  So any mean direction the fit acquires is something it INVENTED to do rho_0's job.

Two fits, differing in one thing:

    plain     phi free
    centred   phi's mean over products subtracted after every step, which removes the global
              direction while leaving pair structure untouched -- ragged.py's --phi-centre

Scored on pair lift, not on the size law, because pairs are what the model is for.  If
centring raises recovered lift, the competition is real and --phi-centre (off since run56) is
the lever.  If it does not, phi is not being diverted and the collapse has another cause.

Run:  python3 s22_compete.py
"""
import math
import time

import numpy as np
import torch

torch.set_default_dtype(torch.float64)

J = 20
KZ = 12
NB = 200000
STEPS = 1500
RANK = 6
TRUE_T = 0.92
B_PAIR = -3.0


def build_mask(Jd):
    idx = torch.arange(2 ** Jd, dtype=torch.int64)
    return ((idx.unsqueeze(1) >> torch.arange(Jd, dtype=torch.int64)) & 1).to(torch.float64)


def energy(mask, nvec, b, PH, r0):
    v = mask @ PH
    return mask @ b + 0.5 * ((v * v).sum(1) - mask @ (PH ** 2).sum(1)) - r0[nvec]


def log_A(mask, nvec, b, PH, Jd):
    E = energy(mask, nvec, b, PH, torch.zeros(Jd + 1, dtype=b.dtype))
    out = torch.full((Jd + 1,), -math.inf, dtype=b.dtype)
    for n in range(Jd + 1):
        sel = nvec == n
        if bool(sel.any()):
            out[n] = torch.logsumexp(E[sel], 0)
    return out


def nb_target(mean, vm, Jd):
    disp = mean / max(vm - 1.0, 1e-6)
    k = np.arange(1, Jd + 1)
    pp = disp / (disp + mean)
    p = np.exp([math.lgamma(x + disp) - math.lgamma(disp) - math.lgamma(x + 1)
                + disp * math.log(pp) + x * math.log(1 - pp) for x in k])
    return torch.as_tensor(np.clip(p, 1e-12, None) / np.clip(p, 1e-12, None).sum())


def erank(PH):
    s = torch.linalg.svdvals(PH)
    return float((s ** 2).sum() ** 2 / (s ** 4).sum().clamp_min(1e-30))


def report(mask_ne, nvec_ne, b, PH, r0, pairs, Jd):
    E = energy(mask_ne, nvec_ne, b, PH, r0)
    p = torch.softmax(E, 0)
    pi = (mask_ne * p.unsqueeze(1)).sum(0)
    law = torch.zeros(Jd + 1, dtype=p.dtype).index_add(0, nvec_ne, p)[1:]
    law = law / law.sum()
    k = torch.arange(1, Jd + 1, dtype=p.dtype)
    mean = float((k * law).sum())
    lifts = []
    for (a, c) in pairs:
        j = float((mask_ne[:, a] * mask_ne[:, c] * p).sum())
        lifts.append(j / max(float(pi[a] * pi[c]), 1e-300))
    return np.array(lifts), law, mean


def main():
    mask = build_mask(J)
    nvec = mask.sum(1).long()
    ne = nvec > 0
    mask_ne, nvec_ne = mask[ne].contiguous(), nvec[ne]
    del mask

    v = math.sqrt(TRUE_T)
    pairs = [(2 * i, 2 * i + 1) for i in range(RANK)]
    PH_t = torch.zeros(J, KZ)
    for i, (a, c) in enumerate(pairs):
        PH_t[a, i] = v
        PH_t[c, i] = v
    b_t = torch.full((J,), -1.0)
    b_t[: 2 * RANK] = B_PAIR                    # the paired products are the rarer ones
    lA = log_A(mask_ne, nvec_ne, b_t, PH_t, J)
    tgt = nb_target(5.0, 2.0, J)
    r0_t = torch.zeros(J + 1)
    r0_t[1:] = lA[1:] - tgt.log()
    r0_t = r0_t - r0_t[1]

    lift_t, law_t, mean_t = report(mask_ne, nvec_ne, b_t, PH_t, r0_t, pairs, J)
    print(f"J = {J}, {RANK} disjoint pairs each at phi'phi = {TRUE_T} in its own direction, "
          f"no global direction")
    print(f"planted: erank {erank(PH_t):.2f}  ||mean(phi)|| {float(PH_t.mean(0).norm()):.4f}  "
          f"mean size {mean_t:.2f}")
    print(f"         pair lift  mean {lift_t.mean():.3f}  worst {lift_t.min():.3f}\n")

    p_t = torch.softmax(energy(mask_ne, nvec_ne, b_t, PH_t, r0_t), 0).numpy()
    draw = np.random.default_rng(0).choice(mask_ne.shape[0], size=NB, p=p_t)
    uniq, inv = np.unique(draw, return_inverse=True)
    red = mask_ne[torch.as_tensor(uniq)].contiguous()
    redn = nvec_ne[torch.as_tensor(uniq)]
    cnt = torch.as_tensor(np.bincount(inv).astype(np.float64))

    print(f"{'fit':>10}{'erank':>8}{'||mean phi||':>14}{'lift mean':>11}{'lift worst':>12}"
          f"{'size KL':>10}{'time':>7}")
    for mode in ("plain", "centred"):
        t0 = time.time()
        bh = torch.full((J,), -0.5, requires_grad=True)
        PH = (torch.randn(J, KZ, generator=torch.Generator().manual_seed(1))
              * 0.1).requires_grad_(True)
        r0 = torch.zeros(J + 1, requires_grad=True)
        opt = torch.optim.Adam([bh, PH, r0], lr=0.05)
        for _ in range(STEPS):
            Ed = energy(red, redn, bh, PH, r0)
            lz = torch.logsumexp(energy(mask_ne, nvec_ne, bh, PH, r0), 0)
            ll = ((Ed - lz) * cnt).sum() / cnt.sum()
            opt.zero_grad()
            (-ll).backward()
            opt.step()
            if mode == "centred":
                with torch.no_grad():
                    PH -= PH.mean(0, keepdim=True)
        with torch.no_grad():
            lift_f, law_f, _ = report(mask_ne, nvec_ne, bh, PH, r0, pairs, J)
            m = law_t > 1e-12
            kl = float((law_t[m] * (law_t[m] / law_f[m].clamp_min(1e-300)).log()).sum())
            print(f"{mode:>10}{erank(PH.detach()):8.2f}{float(PH.mean(0).norm()):14.4f}"
                  f"{lift_f.mean():11.3f}{lift_f.min():12.3f}{kl:10.4f}"
                  f"{time.time()-t0:6.0f}s", flush=True)
    print(f"{'truth':>10}{erank(PH_t):8.2f}{float(PH_t.mean(0).norm()):14.4f}"
          f"{lift_t.mean():11.3f}{lift_t.min():12.3f}{0.0:10.4f}")


if __name__ == "__main__":
    main()
