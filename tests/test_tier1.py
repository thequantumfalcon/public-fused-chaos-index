import json
import tempfile
import unittest
from pathlib import Path

import numpy as np


class Tier1Tests(unittest.TestCase):
    def test_add_quantum_mass_writes_npz_and_manifest(self):
        from fused_chaos_index.tier1.add_quantum_mass import add_quantum_mass_to_catalog_npz

        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            inp = td_path / "in.npz"
            out = td_path / "out.npz"

            rng = np.random.default_rng(0)
            ra = rng.uniform(0.0, 360.0, size=(250,)).astype(np.float64)
            dec = rng.uniform(-30.0, 30.0, size=(250,)).astype(np.float64)
            kappa = rng.normal(size=(250,)).astype(np.float64)

            np.savez(inp, ra=ra, dec=dec, kappa=kappa)

            manifest = add_quantum_mass_to_catalog_npz(
                catalog_npz=inp,
                out_npz=out,
                k=8,
                n_modes=6,
                threshold=5e-7,
                force=False,
            )

            self.assertEqual(manifest.get("experiment"), "Tier-1: Add Quantum Mass to Catalog")
            self.assertEqual(manifest.get("status"), "OK")

            self.assertTrue(out.exists())
            with np.load(out, allow_pickle=True) as z:
                self.assertIn("quantum_mass", z.files)
                self.assertIn("eigenvalues", z.files)
                self.assertIn("dark_percent", z.files)
                self.assertIn("ra", z.files)
                self.assertIn("dec", z.files)
                self.assertIn("kappa", z.files)
                self.assertEqual(z["quantum_mass"].shape[0], 250)
                self.assertEqual(z["eigenvalues"].shape[0], 6)

            mp = out.with_name(out.stem + "_manifest.json")
            self.assertTrue(mp.exists())
            data = json.loads(mp.read_text(encoding="utf-8"))
            self.assertEqual(data.get("experiment"), "Tier-1: Add Quantum Mass to Catalog")
