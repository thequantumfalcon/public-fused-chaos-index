import json
import tempfile
import unittest
from pathlib import Path

from fused_chaos_index.suite import run_public_suite


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


if __name__ == "__main__":
    unittest.main()
