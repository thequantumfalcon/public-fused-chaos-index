"""Run-level manifest generation for CLI commands.

This module provides utilities to create standardized run-level manifests
that document the execution context and outputs of CLI commands.
"""
from __future__ import annotations

import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _now_iso() -> str:
    """Return current UTC timestamp in ISO format."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _package_version() -> str:
    """Get package version from metadata."""
    try:
        from importlib.metadata import version
        return version("public-fused-chaos-index")
    except Exception:
        return "unknown"


def _environment_info() -> dict[str, str]:
    """Collect environment information."""
    env = {
        "python_version": sys.version.replace("\n", " "),
        "python_executable": sys.executable,
        "platform": platform.platform(),
    }
    
    # Add numpy and scipy versions if available
    try:
        import numpy as np
        env["numpy_version"] = np.__version__
    except Exception:
        pass
    
    try:
        import scipy
        env["scipy_version"] = scipy.__version__
    except Exception:
        pass
    
    return env


def create_run_manifest(
    *,
    run_dir: Path,
    command: list[str] | str,
    status: str = "OK",
    outputs: list[dict[str, str]] | None = None,
    additional_fields: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a run-level manifest.json file.
    
    Args:
        run_dir: Path to the run directory where manifest.json will be written
        command: Command that was executed (argv list or string)
        status: Status of the run (OK, ERROR, or SKIP)
        outputs: List of output artifacts, each with 'name', 'path' (relative), and 'type'
        additional_fields: Any additional fields to include in the manifest
        
    Returns:
        The manifest dictionary
    """
    run_id = run_dir.name
    
    manifest = {
        "manifest_schema_version": "1",
        "run_id": run_id,
        "created_utc": _now_iso(),
        "command": command if isinstance(command, str) else " ".join(command),
        "package_version": _package_version(),
        "environment": _environment_info(),
        "status": status,
        "outputs": outputs or [],
    }
    
    if additional_fields:
        manifest.update(additional_fields)
    
    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    
    return manifest


def collect_outputs_from_dir(run_dir: Path, patterns: list[str] | None = None) -> list[dict[str, str]]:
    """Collect output files from run directory.
    
    Args:
        run_dir: Path to the run directory
        patterns: List of glob patterns to match (default: ["*.json", "*.npz", "*.png"])
        
    Returns:
        List of output dictionaries with 'name', 'path', and 'type' keys
    """
    if patterns is None:
        patterns = ["*.json", "*.npz", "*.png"]
    
    outputs = []
    for pattern in patterns:
        for file_path in sorted(run_dir.glob(pattern)):
            if file_path.name == "manifest.json":
                continue  # Skip the run-level manifest itself
            
            relative_path = file_path.relative_to(run_dir)
            suffix = file_path.suffix.lstrip(".")
            
            outputs.append({
                "name": file_path.stem,
                "path": str(relative_path),
                "type": suffix,
            })
    
    return outputs
