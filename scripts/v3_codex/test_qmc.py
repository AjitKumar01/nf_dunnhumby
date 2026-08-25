"""Regression tests for the full-catalogue normaliser path.

Run from the repository root with:
    python -m unittest scripts.v3.test_qmc
"""
import math
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ragged import (RaggedIndex, RaggedModel, esp_bucketed, log_f_ragged,
                    log_f_sparse, gh_grid, set_quad, sparse_prepare,
                    _poly_mul_trunc, _poly_mul_trunc_eager)


torch.set_default_dtype(torch.float64)


def one_row_model(J, Kz, nmax=None):
    nmax = J if nmax is None else nmax
    ix = RaggedIndex(torch.arange(J), torch.zeros(J, dtype=torch.long),
                     torch.tensor([0]), torch.tensor([0]), 1)
    m = RaggedModel(J, 1, 1, K=2, Kz=Kz, nmax=nmax, R=nmax,
                    S=1, Kp=2, Kt=2, Ks=2)
    m.house, m.ctx = torch.tensor([0]), None
    return m, ix


class FullCatalogueQmcTest(unittest.TestCase):
    def test_original_joint_rollout_matches_enumerated_size_and_incidence(self):
        """Corollary 3 must sample the same joint law whose log Z is fitted."""
        torch.manual_seed(23)
        B, J, C, Kz = 2000, 5, 2, 2
        item = torch.arange(J).repeat(B)
        local_row = torch.tensor([0, 0, 1, 1, 1]).repeat(B)
        row_base = torch.arange(B).repeat_interleave(J) * C
        row_of = row_base + local_row
        row_trip = torch.arange(B).repeat_interleave(C)
        row_cat = torch.tensor([0, 1]).repeat(B)
        ix = RaggedIndex(item, row_of, row_trip, row_cat, B)
        m = RaggedModel(J, 1, C, K=2, Kz=Kz, nmax=J, R=J,
                        S=1, Kp=2, Kt=2, Ks=2, seed=5)
        m.house, m.ctx = torch.zeros(B, dtype=torch.long), None
        category = torch.tensor([0, 0, 1, 1, 1])
        with torch.no_grad():
            m.cat_of.copy_(category)
            m.lam.copy_(torch.tensor([-1.1, -0.8, -1.4, -0.5, -1.0]))
            m.phi.copy_(torch.tensor([[0.22, -0.08], [0.16, 0.12],
                                      [-0.11, 0.19], [0.07, 0.24], [0.18, 0.03]]))
            m.rho_c.copy_(torch.tensor([-0.12, 0.07]))
            m.rho_0_free.copy_(torch.tensor([0.0, 0.08, 0.22, 0.45, 0.80]))
        # Positive dense GH weights make stage 1 a directly sampled discrete posterior.
        # At this small interaction strength q=9 agrees with analytic subset weights far
        # below the Monte Carlo tolerance being tested.
        m.quad_z = gh_grid(Kz, 9)

        mask = torch.arange(1, 1 << J)
        X = ((mask[:, None] >> torch.arange(J)) & 1).to(m.lam.dtype)
        n = X.sum(1).to(torch.long)
        summed_phi = X @ m.phi
        pair = 0.5 * (summed_phi.square().sum(1)
                      - X @ m.phi.square().sum(1))
        nc = torch.stack([X[:, category == c].sum(1) for c in range(C)], dim=1)
        energy = (X @ m.lam + pair
                  - (m.rho_c[None, :] * nc * (nc - 1.0) / 2.0).sum(1)
                  - m.rho_0()[n])
        probability = torch.softmax(energy, 0)
        exact_mean = float((probability * n).sum())
        exact_incidence = (probability[:, None] * X).sum(0)

        draws = m.sample(ix, generator=torch.Generator().manual_seed(901))
        sampled_n = torch.tensor([len(s) for s in draws], dtype=m.lam.dtype)
        sampled_incidence = torch.tensor(
            [[float(j in basket) for j in range(J)] for basket in draws]).mean(0)
        self.assertLess(abs(float(sampled_n.mean()) - exact_mean), 0.07)
        self.assertLess(float((sampled_incidence - exact_incidence).abs().max()), 0.035)

    def test_original_joint_value_gradients_size_and_incidence_match_enumeration(self):
        """Audit the complete version-4 law, not only the Gaussian integral's value.

        A correct log Z must give the same size law and parameter derivatives as direct
        enumeration.  In particular d log Z / d b_j is marginal incidence and the sum of
        those incidences is E[n].  This catches an estimator that looks accurate in value
        while optimizing the wrong model.
        """
        torch.manual_seed(113)
        J, C, Kz = 9, 3, 6
        category = torch.arange(J) // 3
        ix = RaggedIndex(torch.arange(J), category,
                         torch.zeros(C, dtype=torch.long), torch.arange(C), 1)
        m = RaggedModel(J, 1, C, K=2, Kz=Kz, nmax=J, R=J,
                        S=1, Kp=2, Kt=2, Ks=2, seed=17)
        m.house, m.ctx = torch.tensor([0]), None
        with torch.no_grad():
            m.cat_of.copy_(category)
            m.lam.copy_(torch.linspace(-2.1, -0.5, J))
            m.phi.normal_()
            m.phi.mul_(0.42 / m.phi.norm(dim=1, keepdim=True))
            m.rho_c.copy_(torch.tensor([-0.10, 0.04, -0.06]))
            n = torch.arange(1, J + 1, dtype=m.lam.dtype)
            m.rho_0_free.copy_(0.035 * n.square())

        # Differentiable exact enumeration of every non-empty subset.
        mask = torch.arange(1, 1 << J)
        X = ((mask[:, None] >> torch.arange(J)) & 1).to(m.lam.dtype)
        n = X.sum(1).to(torch.long)
        b = m.b_flat(ix)
        summed_phi = X @ m.phi
        pair = 0.5 * (summed_phi.square().sum(1)
                      - X @ m.phi.square().sum(1))
        nc = torch.stack([X[:, category == c].sum(1) for c in range(C)], dim=1)
        cat_pen = (m.rho_c[None, :] * nc * (nc - 1.0) / 2.0).sum(1)
        exact_energy = X @ b + pair - cat_pen - m.rho_0()[n]
        exact_lz = torch.logsumexp(exact_energy, 0)
        exact_prob = torch.softmax(exact_energy, 0)
        exact_pn = torch.zeros(J, dtype=m.lam.dtype).index_add_(
            0, n - 1, exact_prob)
        exact_incidence = (exact_prob[:, None] * X).sum(0)
        exact_grad = torch.autograd.grad(
            exact_lz, (m.lam, m.phi, m.rho_c, m.rho_0_free), retain_graph=True)

        set_quad(m, qmc_n=4096, qmc_seed=31, qmc_reps=4, Kz=Kz,
                 probe=Kz, steps=6, chunk=128, size_bands=1, size_steps=3,
                 mode_logtol=8.0, mode_sep=1.0, mix_n=4096)
        q_lz, q_pn = m.log_Z(ix, drop_empty=True, return_size=True)
        q_grad = torch.autograd.grad(
            q_lz, (m.lam, m.phi, m.rho_c, m.rho_0_free))

        self.assertLess(abs(float(q_lz - exact_lz)), 8e-4)
        self.assertLess(float((q_pn[0] - exact_pn).abs().max()), 8e-4)
        self.assertLess(float((q_grad[0] - exact_incidence).abs().max()), 8e-4)
        self.assertLess(abs(float(q_grad[0].sum() - (q_pn[0] *
                            torch.arange(1, J + 1)).sum())), 2e-12)
        self.assertLess(abs(float(exact_incidence.sum() - (exact_pn *
                            torch.arange(1, J + 1)).sum())), 2e-12)
        for got, want in zip(q_grad, exact_grad):
            self.assertLess(float((got - want).abs().max()), 2e-3)

        # Proposition 1 must survive the estimator: a common utility shift delta adds
        # n*delta to every basket, hence d E[n]/d delta = Var(n).  This is the mechanism
        # through which the original joint model obtains aggregate price response.
        eps = 1e-4
        q_var = ((q_pn[0] * torch.arange(1, J + 1).square()).sum()
                 - (q_pn[0] * torch.arange(1, J + 1)).sum().square())
        shifted_means = []
        with torch.no_grad():
            for delta in (-eps, eps):
                m.lam.add_(delta)
                _, shifted_pn = m.log_Z(ix, drop_empty=True, return_size=True)
                shifted_means.append(
                    (shifted_pn[0] * torch.arange(1, J + 1)).sum())
                m.lam.sub_(delta)
        finite_diff = (shifted_means[1] - shifted_means[0]) / (2.0 * eps)
        self.assertLess(abs(float(finite_diff - q_var)), 5e-5)

    def test_exact_additive_normalizer_matches_constant_qmc(self):
        torch.manual_seed(41)
        m, ix = one_row_model(30, 6, nmax=12)
        with torch.no_grad():
            m.lam.normal_(-2.0, 0.25)
            m.phi.zero_()
            m.rho_c.fill_(-0.12)
            m.rho_0_free.copy_(0.02 * torch.arange(1, 13).square())
        set_quad(m, qmc_n=32, qmc_seed=5, qmc_reps=4, Kz=6,
                 probe=-1, steps=2, chunk=8)
        with torch.no_grad():
            q_lz, q_ess, q_pn = m.log_Z(
                ix, drop_empty=True, return_ess=True, return_size=True)
            m._exact_additive = True
            a_lz, a_ess, a_pn = m.log_Z(
                ix, drop_empty=True, return_ess=True, return_size=True)
        self.assertLess(float((a_lz - q_lz).abs().max()), 2e-13)
        self.assertLess(float((a_pn - q_pn).abs().max()), 2e-13)
        self.assertEqual(float(a_ess.min()), 1.0)
        self.assertEqual(float(m._last_qmc_logz_se.max()), 0.0)

        m.zero_grad(set_to_none=True)
        m.log_Z(ix, drop_empty=True, return_size=True)[0].sum().backward()
        self.assertTrue(bool(torch.isfinite(m.lam.grad).all()))
        self.assertTrue(bool(torch.isfinite(m.rho_0_free.grad).all()))

    def test_fused_polynomial_backward_matches_reference(self):
        torch.manual_seed(2)
        # Include a broadcast leading axis: A_const has this shape in the real kernel.
        a = torch.rand(3, 2, 7, requires_grad=True)
        b = torch.rand(1, 2, 5, requires_grad=True)
        ref = _poly_mul_trunc_eager(a, b, 8)
        got = _poly_mul_trunc(a, b, 8)
        self.assertEqual(float((got - ref).abs().max().detach()), 0.0)
        probe = torch.randn_like(got)
        ga = torch.autograd.grad((got * probe).sum(), (a, b), retain_graph=True)
        ra = torch.autograd.grad((ref * probe).sum(), (a, b))
        for x, y in zip(ga, ra):
            self.assertLess(float((x - y).abs().max()), 2e-14)

    def test_esp_covers_rows_above_256(self):
        for n in (300, 1774):
            w = torch.full((1, n), 1.0 / n)
            got = esp_bucketed(w, torch.zeros(n, dtype=torch.long), 1, 4,
                               torch.tensor([n]), torch.arange(n))[0, 0]
            want = torch.tensor([
                1.0,
                1.0,
                (n - 1) / (2 * n),
                (n - 1) * (n - 2) / (6 * n ** 2),
                (n - 1) * (n - 2) * (n - 3) / (24 * n ** 3),
            ])
            self.assertLess(float((got - want).abs().max()), 1e-11)

    def test_sparse_kernel_and_chunked_size_law(self):
        torch.manual_seed(3)
        m, ix = one_row_model(300, 6, nmax=8)
        with torch.no_grad():
            m.lam.normal_(-2.5, 0.2)
            m.phi.normal_(0.0, 0.04)
            m.rho_c.fill_(-0.15)
        z = torch.randn(1, 5, m.Kz)
        with torch.no_grad():
            dense = log_f_ragged(m, z, ix, True)
            sparse = log_f_sparse(m, z, ix, sparse_prepare(m, ix), True)
        self.assertLess(float((dense - sparse).abs().max().detach()), 1e-11)

        set_quad(m, qmc_n=32, qmc_seed=11, qmc_reps=4, Kz=m.Kz,
                 probe=4, steps=2, chunk=0)
        with torch.no_grad():
            whole = m.log_Z(ix, drop_empty=True, return_ess=True, return_size=True)
        m.quad_chunk = 7
        with torch.no_grad():
            chunked = m.log_Z(ix, drop_empty=True, return_ess=True, return_size=True)
        self.assertLess(float((whole[0] - chunked[0]).abs().max()), 1e-12)
        self.assertLess(float((whole[2] - chunked[2]).abs().max()), 1e-12)
        self.assertAlmostEqual(float(chunked[2].sum()), 1.0, places=12)

        m.zero_grad(set_to_none=True)
        m.log_Z(ix, drop_empty=True, return_size=True)[0].sum().backward()
        self.assertTrue(bool(torch.isfinite(m.phi.grad).all()))

    def test_rqmc_matches_exact_subset_normalizer(self):
        torch.manual_seed(7)
        J, Kz = 10, 8
        m, ix = one_row_model(J, Kz)
        with torch.no_grad():
            m.lam.copy_(torch.linspace(-1.7, -0.7, J))
            m.phi.normal_()
            m.phi.mul_(0.55 / m.phi.norm(dim=1, keepdim=True))
            m.rho_c.fill_(0.08)
            n = torch.arange(1, J + 1)
            m.rho_0_free.copy_(0.025 * n ** 2)

        b, rho0 = m.b_flat(ix).detach(), m.rho_0().detach()
        phi, rho_c = m.phi.detach(), m.rho_c.detach()
        terms = []
        for mask in range(1, 1 << J):
            sel = torch.tensor([j for j in range(J) if (mask >> j) & 1])
            p, n = phi[sel], len(sel)
            pair = 0.5 * ((p.sum(0) ** 2).sum() - (p ** 2).sum())
            terms.append(b[sel].sum() + pair
                         - rho_c[0] * n * (n - 1) / 2 - rho0[n])
        exact = torch.logsumexp(torch.stack(terms), dim=0)

        set_quad(m, qmc_n=1024, qmc_seed=9, qmc_reps=4, Kz=Kz,
                 probe=Kz, steps=6, chunk=64)
        with torch.no_grad():
            estimate, _ = m.log_Z(ix, drop_empty=True, return_ess=True)
        self.assertLess(abs(float(estimate - exact)), 3e-3)
        self.assertIsNotNone(m._last_qmc_logz_se)
        self.assertTrue(math.isfinite(float(m._last_qmc_logz_se[0])))

    def test_remote_scaling_identity_frame_and_operator_projection(self):
        # A z-independent scale overflows here: twenty aligned degree-20 weights carry
        # exp(0.96*40) each, so their product is beyond float64.  Per-node rescaling keeps
        # the mathematically finite log polynomial representable.
        m, ix = one_row_model(20, 4)
        with torch.no_grad():
            m.lam.fill_(-1.0)
            m.phi.zero_()
            m.phi[:, 0] = 0.96
            n = torch.arange(1, 21, dtype=m.lam.dtype)
            m.rho_0_free.copy_(0.01 * n.square())
        z = torch.zeros(1, 1, 4)
        z[..., 0] = 40.0
        with torch.no_grad():
            dense = log_f_ragged(m, z, ix, True)
            sparse = log_f_sparse(m, z, ix, sparse_prepare(m, ix), True)
        self.assertTrue(bool(torch.isfinite(sparse).all()))
        self.assertLess(float((dense - sparse).abs().max()), 1e-10)

        # The operator projection caps the catalogue accumulation while retaining all rows.
        torch.manual_seed(19)
        with torch.no_grad():
            m.phi.normal_(0.0, 0.8)
        m.project(phi_max=10.0, op_max=2.0)
        lam = torch.linalg.eigvalsh(m.phi.detach().T @ m.phi.detach())[-1]
        self.assertLessEqual(float(lam), 2.0 + 2e-12)
        self.assertEqual(int((m.phi.norm(dim=1) > 0).sum()), 20)

        set_quad(m, qmc_n=32, qmc_seed=3, qmc_reps=4, Kz=4,
                 probe=-1, steps=2, chunk=32)
        with torch.no_grad():
            cache = sparse_prepare(m, ix)
            _zh, sd, _Q = m._adaptive_frame(ix, True, 2, cache=cache)
        self.assertEqual(float((sd - 1.0).abs().max()), 0.0)

    def test_projection_is_stable_at_repeated_operator_cap(self):
        # Whitening plus the operator cap deliberately produces a repeated singular
        # spectrum.  The tall-matrix SVD used here before run155 could fail to converge on
        # the next update even though the matrix was finite.  Repeated projections must be
        # finite, idempotent and preserve the cap.
        torch.manual_seed(2026)
        m = RaggedModel(545, 2, 1, K=2, Kz=32, nmax=4, R=4,
                        phi_init=0.03).double()
        with torch.no_grad():
            x = torch.randn(545, 32, dtype=torch.float64)
            x -= x.mean(0, keepdim=True)
            q, _ = torch.linalg.qr(x, mode="reduced")
            m.phi.copy_(math.sqrt(2.0) * q)
            for _ in range(4):
                m.project(phi_max=0.96, centre=True, whiten=0.5, op_max=2.0)
            gram = m.phi.T @ m.phi
        self.assertTrue(bool(torch.isfinite(m.phi).all()))
        self.assertLessEqual(float(torch.linalg.eigvalsh(gram)[-1]), 2.0 + 2e-12)
        # Centre changes the deliberately uncentred input on the first pass; subsequent
        # projections should be numerically idempotent at the repeated cap.
        stable = m.phi.detach().clone()
        m.project(phi_max=0.96, centre=True, whiten=0.5, op_max=2.0)
        self.assertLess(float((m.phi.detach() - stable).abs().max()), 2e-12)

    def test_size_stratified_rule_recovers_invisible_remote_basin(self):
        # Construct the exact pathology seen in the failed checkpoint.  The partition has
        # two unit-covariance Gaussian basins, at sizes 2 and 18, whose centres are eight
        # standard deviations apart.  At z=0 the large-size basin is down by about 40 nats,
        # so a zero-start total-mode iteration and even 1024 local nodes never see it.
        J, Kz, p = 20, 4, 0.5
        m, ix = one_row_model(J, Kz)
        desired = torch.full((J + 1,), -20.0)
        desired[2], desired[18] = 0.0, -0.5
        with torch.no_grad():
            m.lam.zero_()
            m.phi.zero_()
            m.phi[:, 0] = p
            m.rho_c.zero_()
            rho = []
            for n in range(1, J + 1):
                base = (math.lgamma(J + 1) - math.lgamma(n + 1)
                        - math.lgamma(J - n + 1) + 0.5 * p * p * n * (n - 1))
                rho.append(base - float(desired[n]))
            m.rho_0_free.copy_(torch.tensor(rho))
        exact = torch.logsumexp(desired[1:], dim=0)

        set_quad(m, qmc_n=1024, qmc_seed=9, qmc_reps=4, Kz=Kz,
                 probe=-1, steps=2, chunk=64)
        with torch.no_grad():
            local = m.log_Z(ix, drop_empty=True)
        self.assertGreater(abs(float(local - exact)), 0.4)

        # The screen takes only three vectorised passes and the integration still uses 32
        # nodes TOTAL (four per mode per replicate), not 32 nodes per mode.
        set_quad(m, qmc_n=32, qmc_seed=9, qmc_reps=4, Kz=Kz,
                 probe=-1, steps=2, chunk=32, size_bands=1, size_steps=3,
                 mode_logtol=4.0, mode_sep=1.0, mix_n=32)
        with torch.no_grad():
            lz, ess, pn = m.log_Z(
                ix, drop_empty=True, return_ess=True, return_size=True)
        self.assertLess(abs(float(lz - exact)), 1e-5)
        self.assertGreater(float(ess), 0.9)
        self.assertEqual(int(m._last_qmc_mode_count[0]), 2)
        self.assertAlmostEqual(float(m._last_qmc_mode_sep[0]), 8.0, places=5)
        expected_n = float((2.0 + 18.0 * math.exp(-0.5)) / (1.0 + math.exp(-0.5)))
        got_n = float((pn * torch.arange(1, J + 1)).sum())
        self.assertAlmostEqual(got_n, expected_n, places=5)

        m.zero_grad(set_to_none=True)
        m.log_Z(ix, drop_empty=True, return_size=True)[0].sum().backward()
        self.assertTrue(bool(torch.isfinite(m.phi.grad).all()))

    def test_size_rule_keeps_the_active_sobol_rotation(self):
        # Regression for run109's broad-shell failure.  The ordinary adaptive rule rotates
        # a finite Sobol block into the Phi'Phi eigenframe, but the first multimode branch
        # accidentally used the raw coordinates.  The Gaussian law is rotation invariant;
        # 32 deterministic Sobol points are not, and one raw scramble dominated log Z.
        torch.manual_seed(29)
        m, ix = one_row_model(24, 4, nmax=12)
        with torch.no_grad():
            m.lam.normal_(-2.0, 0.1)
            m.phi.normal_(0.0, 0.03)
            # Make the leading direction visibly non-axis-aligned.
            v = torch.tensor([0.5, -0.5, 0.5, -0.5])
            m.phi.add_(torch.linspace(-0.08, 0.08, 24)[:, None] * v[None, :])
        set_quad(m, qmc_n=32, qmc_seed=0, qmc_reps=4, Kz=4,
                 probe=-1, steps=2, chunk=32, size_bands=1, size_steps=2,
                 mode_logtol=8.0, mode_sep=100.0, mix_n=64)
        with torch.no_grad():
            cache = sparse_prepare(m, ix)
            z, base, top = m._size_multimode_proposal(ix, True, cache)
            gram = m.phi.T @ m.phi
            ev, Q = torch.linalg.eigh(gram)
            Q = Q[:, torch.argsort(ev, descending=True)]
            x, w = m.quad_a
            recovered = (z - top[:, None, :]) @ Q
            expected_base = (-0.5 * z.square().sum(-1)
                             + 0.5 * x.square().sum(-1)[None, :]
                             + w.log()[None, :])
        self.assertLess(float((recovered - x[None, :, :]).abs().max()), 2e-14)
        self.assertGreater(float((z - top[:, None, :] - x[None, :, :]).abs().max()), 0.1)
        self.assertLess(float((base - expected_base).abs().max()), 2e-14)


if __name__ == "__main__":
    unittest.main()
