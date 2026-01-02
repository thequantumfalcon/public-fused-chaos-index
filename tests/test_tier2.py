import json
import tempfile
import unittest
from pathlib import Path


class Tier2Tests(unittest.TestCase):
    def test_tier2_universality_sweep_writes_manifest(self):
        from fused_chaos_index.tier2.universality_sweep import run_universality_sweep

        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            out_dir = td_path / "out"
            out_dir.mkdir(parents=True, exist_ok=True)

            import numpy as np

            rng = np.random.default_rng(0)
            p1 = td_path / "a.npz"
            p2 = td_path / "b.npz"
            np.savez(p1, quantum_mass=np.abs(rng.normal(size=(100,))).astype(np.float64) + 1e-6, dark_percent=np.float64(12.3))
            np.savez(p2, quantum_mass=np.abs(rng.normal(size=(200,))).astype(np.float64) + 1e-6)

            m = run_universality_sweep(output_dir=out_dir, inputs=[p1, p2], threshold=5e-7, plot=False)
            self.assertIn(m.get("status"), {"OK", "ERROR", "SKIP"})

            mp = out_dir / "tier2_path1_universality_manifest.json"
            self.assertTrue(mp.exists())
            data = json.loads(mp.read_text(encoding="utf-8"))
            self.assertEqual(data.get("experiment"), "Tier-2 Path 1: Ergodic Universality Sweep")
            self.assertTrue((out_dir / "tier2_path1_universality_results.npz").exists())

    def test_tier2_stopping_time_writes_manifest(self):
        from fused_chaos_index.tier2.stopping_time_vs_mass import run_stopping_time_vs_mass

        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            out_dir = td_path / "out"
            out_dir.mkdir(parents=True, exist_ok=True)

            import numpy as np

            rng = np.random.default_rng(0)
            mass = np.abs(rng.normal(size=(2000,))).astype(np.float64) + 1e-6
            inp = td_path / "collatz_like.npz"
            np.savez(inp, quantum_mass=mass)

            m = run_stopping_time_vs_mass(
                output_dir=out_dir,
                input_npz=inp,
                sample_size=500,
                max_steps=200,
                seed=1,
                bootstrap=False,
            )
            self.assertEqual(m.get("status"), "OK")

            mp = out_dir / "tier2_path3_stopping_time_vs_mass_manifest.json"
            self.assertTrue(mp.exists())
            data = json.loads(mp.read_text(encoding="utf-8"))
            self.assertEqual(data.get("experiment"), "Tier-2 Path 3: Stopping Time vs Quantum Mass")
            self.assertTrue((out_dir / "tier2_path3_stopping_time_vs_mass_results.npz").exists())


if __name__ == "__main__":
    unittest.main()
