"""Test run-level manifest creation for CLI commands."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np


class CLIManifestTests(unittest.TestCase):
    """Test that CLI commands create proper run-level manifests."""

    def test_tier2_universality_creates_run_manifest(self):
        """Test that tier2 universality-sweep creates a run-level manifest.json."""
        from fused_chaos_index.cli import main

        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            
            # Create a test input NPZ
            rng = np.random.default_rng(42)
            input_npz = td_path / "test_input.npz"
            np.savez(
                input_npz,
                quantum_mass=np.abs(rng.normal(size=(100,))).astype(np.float64) + 1e-6,
            )
            
            # Run the command
            output_dir = td_path / "output"
            argv = [
                "tier2",
                "universality-sweep",
                "--output-dir",
                str(output_dir),
                "--inputs",
                str(input_npz),
                "--threshold",
                "5e-7",
            ]
            
            exit_code = main(argv)
            self.assertEqual(exit_code, 0)
            
            # Check that a run folder was created
            run_folders = list(output_dir.glob("public_suite_*"))
            self.assertEqual(len(run_folders), 1, "Expected exactly one run folder")
            run_dir = run_folders[0]
            
            # Check that manifest.json exists at run root
            manifest_path = run_dir / "manifest.json"
            self.assertTrue(
                manifest_path.exists(),
                f"manifest.json should exist at {manifest_path}",
            )
            
            # Validate manifest structure
            manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
            
            # Check required top-level keys
            required_keys = [
                "manifest_schema_version",
                "run_id",
                "created_utc",
                "command",
                "package_version",
                "environment",
                "status",
                "outputs",
            ]
            for key in required_keys:
                self.assertIn(
                    key,
                    manifest_data,
                    f"manifest.json should contain '{key}' key",
                )
            
            # Validate specific fields
            self.assertEqual(manifest_data["manifest_schema_version"], "1")
            self.assertEqual(manifest_data["run_id"], run_dir.name)
            self.assertIn(manifest_data["status"], {"OK", "ERROR", "SKIP"})
            
            # Check that command is present and contains the subcommand
            command = manifest_data["command"]
            self.assertIsInstance(command, str)
            self.assertIn("tier2", command)
            self.assertIn("universality-sweep", command)
            
            # Check environment info
            env = manifest_data["environment"]
            self.assertIn("python_version", env)
            self.assertIn("numpy_version", env)
            self.assertIn("scipy_version", env)
            
            # Check outputs list
            outputs = manifest_data["outputs"]
            self.assertIsInstance(outputs, list)
            self.assertGreater(len(outputs), 0, "outputs list should not be empty")
            
            # Validate that each output file exists
            for output in outputs:
                self.assertIn("name", output)
                self.assertIn("path", output)
                self.assertIn("type", output)
                
                output_file = run_dir / output["path"]
                self.assertTrue(
                    output_file.exists(),
                    f"Output file {output_file} should exist",
                )
            
            # Check that per-command manifest still exists
            per_command_manifest = run_dir / "tier2_path1_universality_manifest.json"
            self.assertTrue(
                per_command_manifest.exists(),
                "Per-command manifest should still exist",
            )

    def test_tier2_stopping_time_creates_run_manifest(self):
        """Test that tier2 stopping-time creates a run-level manifest.json."""
        from fused_chaos_index.cli import main

        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            
            # Create a test input NPZ
            rng = np.random.default_rng(42)
            input_npz = td_path / "test_mass.npz"
            np.savez(
                input_npz,
                quantum_mass=np.abs(rng.normal(size=(500,))).astype(np.float64) + 1e-6,
            )
            
            # Run the command
            output_dir = td_path / "output"
            argv = [
                "tier2",
                "stopping-time",
                "--output-dir",
                str(output_dir),
                "--input",
                str(input_npz),
                "--sample-size",
                "100",
                "--max-steps",
                "100",
            ]
            
            exit_code = main(argv)
            self.assertEqual(exit_code, 0)
            
            # Check that a run folder was created
            run_folders = list(output_dir.glob("public_suite_*"))
            self.assertEqual(len(run_folders), 1, "Expected exactly one run folder")
            run_dir = run_folders[0]
            
            # Check that manifest.json exists
            manifest_path = run_dir / "manifest.json"
            self.assertTrue(manifest_path.exists())
            
            # Validate manifest has required keys
            manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertIn("manifest_schema_version", manifest_data)
            self.assertIn("outputs", manifest_data)
            self.assertGreater(len(manifest_data["outputs"]), 0)

    def test_suite_run_creates_run_manifest(self):
        """Test that suite run creates a run-level manifest.json."""
        from fused_chaos_index.cli import main

        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            output_dir = td_path / "output"
            
            # Run a smoke profile (fast and offline)
            argv = [
                "suite",
                "run",
                "--output-dir",
                str(output_dir),
                "--profile",
                "smoke",
            ]
            
            exit_code = main(argv)
            self.assertEqual(exit_code, 0)
            
            # Check that a run folder was created
            run_folders = list(output_dir.glob("public_suite_*"))
            self.assertEqual(len(run_folders), 1, "Expected exactly one run folder")
            run_dir = run_folders[0]
            
            # Check that manifest.json exists
            manifest_path = run_dir / "manifest.json"
            self.assertTrue(
                manifest_path.exists(),
                f"manifest.json should exist at {manifest_path}",
            )
            
            # Validate manifest structure
            manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
            
            # Check required keys
            self.assertIn("manifest_schema_version", manifest_data)
            self.assertIn("command", manifest_data)
            self.assertIn("suite", manifest_data["command"])
            self.assertIn("run", manifest_data["command"])


if __name__ == "__main__":
    unittest.main()
