from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _sha256_file(path: Path, *, block_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(block_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _extract_quantum_mass(data: dict[str, np.ndarray]) -> np.ndarray:
    for key in ("quantum_mass", "mass", "M"):
        if key in data:
            return np.asarray(data[key], dtype=np.float64).reshape(-1)
    raise KeyError("input NPZ missing one of: quantum_mass, mass, M")


def run_plot_16m_discovery(
    *,
    output_dir: Path,
    collatz_npz: Path,
    threshold: float = 5e-7,
    cosmic_target: float = 68.0,
) -> dict[str, Any]:
    """Generate a lightweight plot for the 16M-style Collatz discovery artifact.

    Offline-first and skip-safe:
    - reads one local NPZ
    - attempts to write a PNG if matplotlib is available
    - always writes a results NPZ + JSON manifest

    This is intentionally minimal (no seaborn) to keep base installs light.
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    collatz_npz = Path(collatz_npz)
    if not collatz_npz.exists():
        raise FileNotFoundError(f"Missing collatz NPZ: {collatz_npz}")

    with np.load(collatz_npz, allow_pickle=True) as data:
        d = {k: data[k] for k in data.files}

    mass = _extract_quantum_mass(d)
    dark_percent_stored: float | None
    try:
        dark_percent_stored = float(np.asarray(d.get("dark_percent"))) if "dark_percent" in d else None
    except Exception:
        dark_percent_stored = None

    dark_percent_computed = float(100.0 * np.mean(mass < float(threshold))) if mass.size else float("nan")
    dark_percent_used = dark_percent_stored if dark_percent_stored is not None else dark_percent_computed

    evals = np.asarray(d.get("eigenvalues", np.array([], dtype=np.float64)), dtype=np.float64).reshape(-1)
    evals_sorted = np.sort(evals) if evals.size else np.array([], dtype=np.float64)

    first_gap = float("nan")
    gap_ratio = float("nan")
    if evals_sorted.size >= 2:
        first_gap = float(evals_sorted[1] - evals_sorted[0])
        if evals_sorted[0] != 0.0:
            gap_ratio = float(first_gap / evals_sorted[0])

    issues: list[str] = []
    plot_png: str | None = None

    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

        ax0 = axes[0]
        eps = np.finfo(np.float64).tiny
        ax0.hist(np.log10(mass + eps), bins=60)
        ax0.axvline(np.log10(float(threshold)), linestyle="--", linewidth=2)
        ax0.set_title("Quantum mass distribution")
        ax0.set_xlabel("log10(mass)")
        ax0.set_ylabel("count")

        ax1 = axes[1]
        if evals_sorted.size:
            ax1.plot(np.arange(evals_sorted.size), evals_sorted, marker="o")
            ax1.set_yscale("log")
            ax1.set_title("Low-energy spectrum")
            ax1.set_xlabel("mode")
            ax1.set_ylabel("eigenvalue")
        else:
            ax1.axis("off")
            ax1.text(0.05, 0.6, "No eigenvalues in NPZ", transform=ax1.transAxes)

        fig.suptitle(
            f"dark%={dark_percent_used:.2f} (target {cosmic_target:.1f}%)  gap_ratio={gap_ratio if np.isfinite(gap_ratio) else float('nan'):.6g}",
            fontsize=10,
        )
        fig.tight_layout()

        out_png = output_dir / "tier2_plot_16m_discovery.png"
        fig.savefig(out_png, dpi=200)
        plt.close(fig)
        plot_png = str(out_png)
    except Exception as e:
        issues.append(f"plot: {type(e).__name__}: {e}")
        plot_png = None

    out_npz = output_dir / "tier2_plot_16m_discovery_results.npz"
    np.savez(
        out_npz,
        collatz_file=str(collatz_npz),
        collatz_sha256=_sha256_file(collatz_npz),
        threshold=np.float64(threshold),
        cosmic_target=np.float64(cosmic_target),
        dark_percent_stored=(np.float64(dark_percent_stored) if dark_percent_stored is not None else np.float64(np.nan)),
        dark_percent_computed=np.float64(dark_percent_computed),
        dark_percent_used=np.float64(dark_percent_used),
        eigenvalues=evals_sorted,
        first_gap=np.float64(first_gap),
        gap_ratio=np.float64(gap_ratio),
        plot_png=(plot_png if plot_png is not None else ""),
        issues=np.array(issues, dtype=object),
    )

    status = "OK" if plot_png is not None else "SKIP"

    manifest = {
        "experiment": "Tier-2: Plot 16M Discovery Figure",
        "time_utc": _now_iso(),
        "status": status,
        "inputs": {
            "collatz": str(collatz_npz),
            "collatz_sha256": _sha256_file(collatz_npz),
        },
        "params": {
            "threshold": float(threshold),
            "cosmic_target": float(cosmic_target),
        },
        "results": {
            "dark_percent_stored": dark_percent_stored,
            "dark_percent_computed": float(dark_percent_computed),
            "dark_percent_used": float(dark_percent_used),
            "spectrum": {
                "k": int(evals_sorted.size),
                "first_gap": (None if not np.isfinite(first_gap) else float(first_gap)),
                "gap_ratio": (None if not np.isfinite(gap_ratio) else float(gap_ratio)),
            },
        },
        "outputs": {
            "results_npz": str(out_npz),
            "plot_png": plot_png,
        },
        "issues": issues,
    }

    (output_dir / "tier2_plot_16m_discovery_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    return manifest
