import unittest
from unittest.mock import patch

import numpy as np
import torch
from study import cost, logged_data, plan, train_ensemble


class MPCTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.X, cls.y = logged_data(1, 60)
        cls.models = train_ensemble(cls.X, cls.y, members=2, epochs=10)

    def test_data(self):
        X, y = logged_data(1, 60)
        np.testing.assert_array_equal(X, self.X)
        np.testing.assert_array_equal(y, self.y)

    def test_no_plant_access(self):
        with patch("study.plant", side_effect=AssertionError("information leakage")):
            u = plan(self.models, 0.5, [0.4, 0.4], population=12, particles=3)
        self.assertTrue(0 <= u <= 1)

    def test_reproducible_plan(self):
        a = plan(self.models, 0.3, [0.4, 0.5], seed=7)
        b = plan(self.models, 0.3, [0.4, 0.5], seed=7)
        self.assertEqual(a, b)

    def test_uncertainty(self):
        for m in self.models:
            _, v = m(torch.zeros((4, 3)))
            self.assertTrue(torch.isfinite(v).all())

    def test_cost(self):
        self.assertGreater(cost(-1, 0), cost(1, 0))

    def test_invalid(self):
        with self.assertRaises(ValueError):
            train_ensemble([[1, 2]], [1])
        with self.assertRaises(ValueError):
            plan([], 0, [0.1])

    def test_risk_actions(self):
        for risk in (0, 1, 5):
            self.assertTrue(0 <= plan(self.models, 0.1, [0.5], risk=risk) <= 1)


if __name__ == "__main__":
    unittest.main()
