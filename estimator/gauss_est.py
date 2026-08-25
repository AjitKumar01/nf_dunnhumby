"""A quadrature-free normaliser for the version-4 set law.

THEORY.  Hubbard-Stratonovich rewrites the pair term as a Gaussian integral, and the code
has been spending its budget evaluating that integral on 8-512 nodes.  But

    f(z) = sum_S W(S) e^{z'v_S},   v_S = sum_{j in S} phi_j,
    W(S) = exp( sum_{j in S}(b_j - ||phi_j||^2/2) - rho_0(n) - sum_c rho_c C(n_c,2) )

so log f is exactly the CUMULANT GENERATING FUNCTION of v_S under W, and

    Z = E_z[f(z)] = f(0) * E_{S~W}[ exp(||v_S||^2 / 2) ].

The Gaussian integral is a change of variables, not the difficulty.  The real quantity is
ONE expectation of exp(||v||^2/2) over the law of the basket interaction vector.  If v_S is
approximately N(a, B),

    log Z  =  log f(0)  -  1/2 log|I - B|  +  1/2 a'(I-B)^{-1} a                      (*)

exactly, with a = grad log f(0) = E[v_S] and B = Hess log f(0) = Cov(v_S).  No nodes.

VALIDITY.  (*) needs I - B positive definite, i.e. lambda_max(Cov v) < 1 -- the same
condition the trainer already logs as lam_max (run155: 0.177).  It also explains why log Z
was measured flat from 8 to 512 nodes: the integrand is nearly quadratic, so extra nodes
bought nothing.

COST.  One ESP pass for f(0), one backward for a, and B either exactly (Kz Hessian-vector
products) or by the mean-field form Phi' diag(pi(1-pi)) Phi at O(J Kz^2).  The latter is
what makes this O(1) in node count.
"""
import math
import torch


def esp_weighted(u, cat, rho_c, rho0, nmax):
    """A_n for all n, with per-category pair penalties.  Exact, O(J * nmax)."""
    C = int(cat.max().item()) + 1 if cat.numel() else 1
    poly = torch.zeros(nmax + 1, dtype=u.dtype, device=u.device)
    poly[0] = 1.0
    for c in range(C):
        sel = cat == c
        if not bool(sel.any()):
            continue
        uc = u[sel]
        e = torch.zeros(len(uc) + 1, dtype=u.dtype, device=u.device)
        e[0] = 1.0
        for x in uc:
            e = e + x * torch.cat([torch.zeros(1, dtype=u.dtype, device=u.device), e[:-1]])
        r = torch.arange(len(e), dtype=u.dtype, device=u.device)
        e = e * torch.exp(-rho_c[c] * r * (r - 1) / 2.0)
        new = torch.zeros_like(poly)
        for k in range(min(len(e), nmax + 1)):
            if float(e[k]) == 0.0:
                continue
            new[k:] = new[k:] + e[k] * poly[: nmax + 1 - k]
        poly = new
    return poly * torch.exp(-rho0[: nmax + 1])


def log_f(z, b, phi, cat, rho_c, rho0, nmax, drop_empty=True):
    """log f(z) -- exact given z."""
    u = torch.exp(b - 0.5 * (phi ** 2).sum(-1) + phi @ z)
    A = esp_weighted(u, cat, rho_c, rho0, nmax)
    A = A[1:] if drop_empty else A
    return torch.log(A.clamp_min(1e-300)).logsumexp(0)


def log_Z_gauss(b, phi, cat, rho_c, rho0, nmax, mode="exact", drop_empty=True):
    """log Z by the Gaussian (second-cumulant) closed form -- NO quadrature nodes.

    mode="exact": B by Kz Hessian-vector products (Kz backward passes).
    mode="mf":    B by the mean-field form, one backward pass total.
    """
    Kz = phi.shape[1]
    z = torch.zeros(Kz, dtype=phi.dtype, device=phi.device, requires_grad=True)
    g0 = log_f(z, b, phi, cat, rho_c, rho0, nmax, drop_empty)
    a = torch.autograd.grad(g0, z, create_graph=(mode == "exact"))[0]
    if mode == "exact":
        rows = []
        for k in range(Kz):
            e = torch.zeros(Kz, dtype=phi.dtype, device=phi.device); e[k] = 1.0
            rows.append(torch.autograd.grad(a @ e, z, retain_graph=True)[0])
        B = torch.stack(rows)
    else:
        # pi_j = P(j in S): one backward pass in b, then Cov(v) ~ Phi' diag(pi(1-pi)) Phi
        bb = b.detach().clone().requires_grad_(True)
        gb = log_f(torch.zeros(Kz, dtype=phi.dtype), bb, phi, cat, rho_c, rho0, nmax, drop_empty)
        pi = torch.autograd.grad(gb, bb)[0]
        B = phi.T @ (phi * (pi * (1 - pi)).unsqueeze(-1))
    B = 0.5 * (B + B.T)
    I = torch.eye(Kz, dtype=phi.dtype, device=phi.device)
    M = I - B
    ev = torch.linalg.eigvalsh(M)
    if float(ev.min()) <= 1e-8:                       # expansion invalid: lambda_max(B) >= 1
        return float("nan"), float(1.0 - ev.min())
    sol = torch.linalg.solve(M, a.detach())
    lz = (g0.detach() - 0.5 * torch.logdet(M) + 0.5 * (a.detach() @ sol))
    return float(lz), float(1.0 - ev.min())


def log_Z_cv(b, phi, cat, rho_c, rho0, nmax, n_nodes=8, seed=0, mode="mf",
             drop_empty=True, reps=2):
    """log Z by Laplace-tilted importance sampling -- EXACT (unbiased in Z), few nodes.

    The closed form alone is only good while lambda_max(Cov v) stays small, and lambda_max
    grows precisely as phi learns, so it cannot be the answer on its own.  But the quadratic
    q(z) = exp(g0 + a'z + z'Bz/2) has BOTH a closed-form Gaussian integral and a closed-form
    tilted measure,

        Z = E_z[f] = Z_gauss * E_{z ~ N(m, V)}[ f(z)/q(z) ],
        V = (I - B)^{-1},   m = V a,

    and f/q = 1 to second order by construction.  So the residual integrand is flat and a
    handful of nodes suffices, while the estimator stays unbiased in Z however large
    lambda_max grows -- the closed form supplies the shape, the nodes supply the truth.
    """
    Kz = phi.shape[1]
    z0 = torch.zeros(Kz, dtype=phi.dtype, device=phi.device, requires_grad=True)
    g0 = log_f(z0, b, phi, cat, rho_c, rho0, nmax, drop_empty)
    a = torch.autograd.grad(g0, z0, create_graph=(mode == "exact"))[0]
    if mode == "exact":
        B = torch.stack([torch.autograd.grad(a[k], z0, retain_graph=True)[0] for k in range(Kz)])
    else:
        bb = b.detach().clone().requires_grad_(True)
        gb = log_f(torch.zeros(Kz, dtype=phi.dtype), bb, phi, cat, rho_c, rho0, nmax, drop_empty)
        pi = torch.autograd.grad(gb, bb)[0]
        B = phi.T @ (phi * (pi * (1 - pi)).unsqueeze(-1))
    B = 0.5 * (B + B.T)
    a, g0 = a.detach(), g0.detach()
    I = torch.eye(Kz, dtype=phi.dtype, device=phi.device)
    M = I - B
    ev = torch.linalg.eigvalsh(M)
    if float(ev.min()) <= 1e-6:
        M = M + (1e-6 - float(ev.min())) * I        # keep the PROPOSAL valid
    # q and the tilt MUST use the same matrix.  The identity is
    #     Z = [int q phi_I] * E_{N(m,V)}[f/q],   V = M^{-1}, m = V a,
    # which holds for ANY B_eff with I - B_eff > 0 -- but only if the SAME B_eff appears in
    # q(z) and in the proposal.  Patching M while leaving q on the unpatched B breaks the
    # identity and biases the result exactly when the safeguard fires.
    B_eff = I - M
    V = torch.linalg.inv(M)
    m = V @ a
    L = torch.linalg.cholesky(V)
    log_Zq = g0 - 0.5 * torch.logdet(M) + 0.5 * (a @ (V @ a))
    acc = []
    for r in range(reps):
        eng = torch.quasirandom.SobolEngine(Kz, scramble=True, seed=seed + 7919 * r)
        u = eng.draw(max(n_nodes // reps, 1)).double().clamp(1e-12, 1 - 1e-12)
        x = torch.erfinv(2 * u - 1) * math.sqrt(2.0)
        lr_ = []
        for i in range(x.shape[0]):
            z = m + L @ x[i].to(phi.dtype)
            lf = log_f(z, b, phi, cat, rho_c, rho0, nmax, drop_empty)
            lq = g0 + a @ z + 0.5 * (z @ (B_eff @ z))
            lr_.append(lf - lq)
        acc.append(torch.logsumexp(torch.stack(lr_), 0) - math.log(x.shape[0]))
    return float(log_Zq + (torch.logsumexp(torch.stack(acc), 0) - math.log(reps)))
