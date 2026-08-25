"""Regression tests for fresh-run initialization."""
import unittest

import numpy as np
import torch

from fit import (observed_composition_loglik, observed_factored_loglik,
                 optimizer_parameter_groups,
                 popularity_logits)
from ragged import RaggedModel


torch.set_default_dtype(torch.float64)


class InitializationTest(unittest.TestCase):
    def test_popularity_uses_incidence_divided_by_assortment_exposure(self):
        # Two training trips at two stores.  Item 2 is offered twice and never selected;
        # item 0 is offered once and selected once.  The centred gauge must preserve their
        # smoothed rate ratio: (1.5/2) / (0.5/3) = 4.5.
        D = dict(n_item=4, n_cat=1, n_store=2,
                 line_ptr=np.array([0, 2, 3]),
                 line_item=np.array([0, 1, 1]),
                 trip_store=np.array([0, 1]),
                 store_cat_ptr=np.array([0, 3, 6]),
                 store_items=np.array([0, 1, 2, 1, 2, 3]))
        got = popularity_logits(D, np.array([0, 1]))
        self.assertAlmostEqual(float(torch.exp(got[0] - got[2])), 4.5, places=12)
        self.assertAlmostEqual(float(got.mean()), 0.0, places=12)

    def test_low_taste_scale_changes_only_taste_initialization(self):
        large = RaggedModel(7, 5, 2, K=3, Kz=4, nmax=5, R=2,
                            seed=9, taste_init=0.3)
        small = RaggedModel(7, 5, 2, K=3, Kz=4, nmax=5, R=2,
                            seed=9, taste_init=0.03)
        self.assertLess(float((small.alpha - 0.1 * large.alpha).abs().max()), 1e-15)
        self.assertLess(float((small.theta - 0.1 * large.theta).abs().max()), 1e-15)
        self.assertTrue(torch.equal(small.phi, large.phi))

    def test_product_intercept_has_independent_learning_rate(self):
        model = RaggedModel(7, 5, 2, K=3, Kz=4, nmax=5, R=2, seed=9)
        groups = optimizer_parameter_groups(model, lr=0.002, lam_lr_scale=0.05)
        self.assertEqual([group["group_name"] for group in groups], ["main", "lam"])
        self.assertEqual([group["lr"] for group in groups], [0.002, 0.0001])
        self.assertIs(groups[1]["params"][0], model.lam)
        self.assertNotIn(id(model.lam), {id(p) for p in groups[0]["params"]})

    def test_zero_product_intercept_rate_is_a_valid_frozen_group(self):
        model = RaggedModel(7, 5, 2, K=3, Kz=4, nmax=5, R=2, seed=9)
        groups = optimizer_parameter_groups(model, lr=0.002, lam_lr_scale=0.0)
        opt = torch.optim.Adam(groups, weight_decay=1e-5)
        before = model.lam.detach().clone()
        sum(p.square().sum() for p in model.parameters()).backward()
        opt.step()
        self.assertTrue(torch.equal(before, model.lam))

    def test_conditional_composition_subtracts_observed_size_probability(self):
        joint = torch.tensor([-4.0, -8.0])
        pn = torch.tensor([[0.2, 0.3, 0.5], [0.6, 0.3, 0.1]])
        line_trip = torch.tensor([0, 0, 1])  # observed sizes two and one
        got = observed_composition_loglik(joint, pn, line_trip)
        want = joint - torch.log(torch.tensor([0.3, 0.6]))
        self.assertTrue(torch.allclose(got, want, atol=0, rtol=0))

    def test_factored_likelihood_replaces_internal_size_probability(self):
        joint = torch.tensor([-4.0, -8.0])
        internal = torch.tensor([[0.2, 0.3, 0.5], [0.6, 0.3, 0.1]])
        external_log = torch.log(torch.tensor([0.4, 0.35, 0.25]))
        line_trip = torch.tensor([0, 0, 1])
        got = observed_factored_loglik(joint, internal, line_trip, external_log)
        want = torch.tensor([-4.0, -8.0]) - torch.log(torch.tensor([0.3, 0.6])) \
            + torch.log(torch.tensor([0.35, 0.4]))
        self.assertTrue(torch.allclose(got, want, atol=0, rtol=0))

    def test_category_pair_feature_uses_original_quadratic_on_full_support(self):
        model = RaggedModel(4, 3, 2, K=2, Kz=2, nmax=120, R=120, seed=4)
        count = torch.tensor([0.0, 1.0, 2.0, 22.0, 23.0, 24.0, 120.0])
        got = model.pair_feature(count)
        self.assertTrue(torch.equal(got, count * (count - 1) / 2))
        inc = model.pair_increment(torch.tensor([22.0, 23.0, 119.0]))
        self.assertTrue(torch.equal(inc, torch.tensor([22.0, 23.0, 119.0])))


if __name__ == "__main__":
    unittest.main()
