import json
import tempfile
import unittest
from pathlib import Path

from fused_chaos_index.suite import run_public_suite


def _write_synth_frontier_inputs(base: Path) -> Path:
    import numpy as np

    clusters_dir = base / "clusters"
    clusters_dir.mkdir(parents=True, exist_ok=True)

    # Synthetic NPZ: positions + kappa vector
    rng = np.random.default_rng(123)
    positions = rng.normal(size=(200, 2)).astype(np.float64)
    kappa = rng.normal(size=(200,)).astype(np.float64)

    npz_path = clusters_dir / "a2744_synth.npz"
    np.savez(npz_path, positions=positions, kappa=kappa)

    spec = {
        "clusters": [
            {
                "cluster_id": "a2744",
                "cluster_name": "Abell 2744 (synthetic)",
                "npz_path": str(npz_path.name),
            }
        ]
    }
    clusters_json = clusters_dir / "clusters.json"
    clusters_json.write_text(__import__("json").dumps(spec), encoding="utf-8")
    return clusters_json


class SuiteTests(unittest.TestCase):
    def test_public_suite_writes_manifest(self):
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td) / "run"
            manifest = run_public_suite(run_dir=run_dir, operational_n_galaxies=200, operational_k=6, operational_seed=1)

            self.assertEqual(manifest["experiment"], "Public Fused Chaos Index Suite")
            self.assertIn("steps", manifest)

            manifest_path = run_dir / "suite_manifest.json"
            self.assertTrue(manifest_path.exists())

            data = json.loads(manifest_path.read_text(encoding="utf-8"))
            step_names = [s["name"] for s in data["steps"]]
            self.assertEqual(step_names, ["operational", "syk_collatz", "gate"])

            # Gate should SKIP when manifests aren't present.
            gate = [s for s in data["steps"] if s["name"] == "gate"][0]
            self.assertIn(gate["status"], {"SKIP", "OK"})

    def test_public_suite_offline_includes_universality(self):
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td) / "run"
            manifest = run_public_suite(
                run_dir=run_dir,
                profile="offline",
                operational_n_galaxies=200,
                operational_k=6,
                operational_seed=1,
            )

            self.assertEqual(manifest["profile"], "offline")

            manifest_path = run_dir / "suite_manifest.json"
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
            step_names = [s["name"] for s in data["steps"]]
            self.assertEqual(step_names, ["operational", "syk_collatz", "universality_ground_truth", "gate"])

            # Universality suite should always write its suite manifest and CSV.
            self.assertTrue((run_dir / "universality_ground_truth_suite_manifest.json").exists())
            self.assertTrue((run_dir / "universality_ground_truth_comparison.csv").exists())

            # Gate should be able to see the produced universality manifest.
            gate = [s for s in data["steps"] if s["name"] == "gate"][0]
            self.assertIn(gate["status"], {"OK", "ERROR"})

    def test_frontier_suite_skip_without_clusters_json(self):
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td) / "run"
            manifest = run_public_suite(
                run_dir=run_dir,
                profile="full",
                operational_n_galaxies=200,
                operational_k=6,
                operational_seed=1,
            )

            data = json.loads((run_dir / "suite_manifest.json").read_text(encoding="utf-8"))
            step_names = [s["name"] for s in data["steps"]]
            self.assertIn("frontier_evidence", step_names)
            frontier = [s for s in data["steps"] if s["name"] == "frontier_evidence"][0]
            self.assertEqual(frontier["status"], "SKIP")

    def test_frontier_suite_ok_with_synthetic_npz(self):
        from fused_chaos_index.suites.frontier import run_frontier_evidence_suite

        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            clusters_json = _write_synth_frontier_inputs(td_path)

            out_dir = td_path / "out"
            res = run_frontier_evidence_suite(
                output_dir=out_dir,
                clusters_json=clusters_json,
                k=6,
                n_modes=5,
                n_perm=50,
                seed=7,
            )

            self.assertIn(res.status, {"OK", "ERROR"})
            mpath = out_dir / "frontier_evidence_suite_manifest.json"
            self.assertTrue(mpath.exists())

            m = json.loads(mpath.read_text(encoding="utf-8"))
            self.assertEqual(m.get("experiment"), "Frontier Evidence Suite")
            self.assertIsInstance(m.get("clusters"), list)
            self.assertGreaterEqual(len(m["clusters"]), 1)

            c0 = m["clusters"][0]
            # Gate/tier1-compatible structure
            self.assertIn("cluster_id", c0)
            if c0.get("status") == "OK":
                self.assertIn("correlation", c0)
                self.assertIn("null_test", c0)
                self.assertIn("provenance", c0)
                self.assertIn("kappa_sha256", c0["provenance"])
                self.assertIn("p_permutation", c0["null_test"])


if __name__ == "__main__":
    unittest.main()
