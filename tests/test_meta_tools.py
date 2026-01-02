import unittest

import numpy as np

from fused_chaos_index.meta_tools import ablation_sensitivity, combine_evidence_heuristic, keyword_risk_audit


class MetaToolsTests(unittest.TestCase):
    def test_keyword_audit(self):
        status, matched = keyword_risk_audit("this is about nuclear stuff")
        self.assertEqual(status, "Flagged")
        self.assertIn("nuclear", matched)

    def test_combine_evidence_range(self):
        s = combine_evidence_heuristic([0.9, 0.8, 0.7], prior=0.5)
        self.assertTrue(0.0 <= s <= 1.0)

    def test_ablation_runs(self):
        x = np.arange(50, dtype=np.float64)
        y = x + 0.01 * np.random.default_rng(0).normal(size=50)
        r = ablation_sensitivity(x, y, n_trials=20, method="spearman", seed=0)
        self.assertTrue(np.isfinite(r.baseline))


if __name__ == "__main__":
    unittest.main()
