import unittest

from fused_chaos_index import StreamlinedFCIPipeline, compute_syk_collatz_fci_constant


class SmokeTests(unittest.TestCase):
    def test_syk_collatz_analytic(self):
        r = compute_syk_collatz_fci_constant(method="analytic")
        self.assertAlmostEqual(r.C, 3.72 * -1.23, places=12)

    def test_operational_pipeline_runs(self):
        pipeline = StreamlinedFCIPipeline(k_neighbors=6, seed=1)
        out = pipeline.run(n_galaxies=300, n_eigenstates=20)
        self.assertIn("fci", out)
        self.assertTrue(out["fci"].fci_normalized >= 0.0)


if __name__ == "__main__":
    unittest.main()
