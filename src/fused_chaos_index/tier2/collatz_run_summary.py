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


def _safe_float(x: Any) -> float | None:
    try:
        return float(np.asarray(x))
    except Exception:
        return None


def _extract_quantum_mass(data: dict[str, np.ndarray]) -> np.ndarray:
    for key in ("quantum_mass", "mass", "M"):
        if key in data:
            return np.asarray(data[key], dtype=np.float64).reshape(-1)
    raise KeyError("input NPZ missing one of: quantum_mass, mass, M")


def _percentiles(values: np.ndarray, ps: list[int]) -> dict[str, float]:
    if values.size == 0:
        return {f"p{p:02d}": float("nan") for p in ps}
    out: dict[str, float] = {}
    for p in ps:
        out[f"p{p:02d}"] = float(np.percentile(values, p))
    return out


def _spectrum_summary(evals: np.ndarray) -> dict[str, Any]:
    x = np.asarray(evals, dtype=np.float64).reshape(-1)
    if x.size == 0:
        return {"k": 0, "eigenvalues": None, "gap": None, "gap_ratio": None}

    x = np.sort(x)
    gap: float | None = None
    gap_ratio: float | None = None

    if x.size >= 2:
        gap = float(x[1] - x[0])
        if x[0] != 0.0:
            gap_ratio = float(gap / x[0])

    return {
        "k": int(x.size),
        "eigenvalues": x.tolist(),
        "gap": gap,
        "gap_ratio": gap_ratio,
        "min": float(x[0]),
        "max": float(x[-1]),
    }


def run_collatz_run_summary(
    *,
    output_dir: Path,
    collatz_npz: Path,
    baseline_npz: Path | None = None,
    threshold: float = 5e-7,
    cosmic_target: float = 68.0,
    runtime_seconds: float | None = None,
) -> dict[str, Any]:
    """Summarize a Collatz run artifact (NPZ) into manuscript-ready stats.

    Offline-first: consumes local NPZ files only and writes a JSON manifest + NPZ results.

    Required inputs:
    - collatz_npz: must contain `quantum_mass` (or `mass`/`M`)

    Optional inputs:
    - baseline_npz: second NPZ for comparison (e.g., 10M baseline)

    Notes:
    - If `dark_percent` exists in NPZ, it is recorded.
    - If not, dark% is computed from the mass vector and `threshold`.
    - If `eigenvalues` exists, the spectral gap and gap ratio are computed.
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    collatz_npz = Path(collatz_npz)
    baseline_npz = Path(baseline_npz) if baseline_npz is not None else None

    if not collatz_npz.exists():
        raise FileNotFoundError(f"Missing collatz_npz: {collatz_npz}")

    collatz_data = np.load(collatz_npz, allow_pickle=True)
    collatz_dict = {k: collatz_data[k] for k in collatz_data.files}
    mass = _extract_quantum_mass(collatz_dict)

    dark_percent_stored = _safe_float(collatz_dict.get("dark_percent"))
    dark_percent_computed = float(100.0 * np.mean(mass < float(threshold))) if mass.size else float("nan")

    dark_percent_used = dark_percent_stored if dark_percent_stored is not None else dark_percent_computed
    deviation_abs = float(abs(float(dark_percent_used) - float(cosmic_target)))
    deviation_rel_pct = float((deviation_abs / float(cosmic_target)) * 100.0) if cosmic_target != 0.0 else float("nan")

    mass_stats = {
        "n": int(mass.size),
        "mean": float(np.mean(mass)) if mass.size else float("nan"),
        "median": float(np.median(mass)) if mass.size else float("nan"),
        "std": float(np.std(mass)) if mass.size else float("nan"),
        "min": float(np.min(mass)) if mass.size else float("nan"),
        "max": float(np.max(mass)) if mass.size else float("nan"),
        "percentiles": _percentiles(mass, [10, 25, 50, 75, 90, 95, 99]),
        "threshold": float(threshold),
        "dark_candidates": int(np.sum(mass < float(threshold))) if mass.size else 0,
        "dark_percent_computed": float(dark_percent_computed),
        "dark_percent_stored": dark_percent_stored,
        "dark_percent_used": float(dark_percent_used),
    }

    spectrum = _spectrum_summary(collatz_dict.get("eigenvalues", np.array([], dtype=np.float64)))

    baseline_summary: dict[str, Any] | None = None
    if baseline_npz is not None and baseline_npz.exists():
        b = np.load(baseline_npz, allow_pickle=True)
        bdict = {k: b[k] for k in b.files}
        bmass = _extract_quantum_mass(bdict)
        b_dark_percent_stored = _safe_float(bdict.get("dark_percent"))
        b_dark_percent_computed = float(100.0 * np.mean(bmass < float(threshold))) if bmass.size else float("nan")
        b_dark_used = b_dark_percent_stored if b_dark_percent_stored is not None else b_dark_percent_computed
        baseline_summary = {
            "file": str(baseline_npz),
            "sha256": _sha256_file(baseline_npz),
            "n": int(bmass.size),
            "dark_percent_stored": b_dark_percent_stored,
            "dark_percent_computed": float(b_dark_percent_computed),
            "dark_percent_used": float(b_dark_used),
        }

    out_npz = output_dir / "tier2_collatz_run_summary_results.npz"
    np.savez(
        out_npz,
        collatz_file=str(collatz_npz),
        collatz_sha256=_sha256_file(collatz_npz),
        baseline_file=(str(baseline_npz) if baseline_npz is not None else ""),
        baseline_sha256=(_sha256_file(baseline_npz) if baseline_npz is not None and baseline_npz.exists() else ""),
        threshold=np.float64(threshold),
        cosmic_target=np.float64(cosmic_target),
        runtime_seconds=(np.float64(runtime_seconds) if runtime_seconds is not None else np.float64(np.nan)),
        dark_percent_used=np.float64(dark_percent_used),
        deviation_abs=np.float64(deviation_abs),
        deviation_rel_pct=np.float64(deviation_rel_pct),
        mass_stats=np.array([mass_stats], dtype=object),
        spectrum=np.array([spectrum], dtype=object),
        baseline=np.array([baseline_summary], dtype=object) if baseline_summary is not None else np.array([], dtype=object),
    )

    manifest = {
        "experiment": "Tier-2: Collatz Run Summary (Manuscript Stats)",
        "time_utc": _now_iso(),
        "status": "OK",
        "inputs": {
            "collatz": str(collatz_npz),
            "collatz_sha256": _sha256_file(collatz_npz),
            "baseline": (str(baseline_npz) if baseline_npz is not None else None),
            "baseline_sha256": (_sha256_file(baseline_npz) if baseline_npz is not None and baseline_npz.exists() else None),
        },
        "params": {
            "threshold": float(threshold),
            "cosmic_target": float(cosmic_target),
            "runtime_seconds": (float(runtime_seconds) if runtime_seconds is not None else None),
        },
        "results": {
            "dark_percent_used": float(dark_percent_used),
            "deviation_abs": float(deviation_abs),
            "deviation_rel_pct": float(deviation_rel_pct),
            "mass_stats": mass_stats,
            "spectrum": spectrum,
            "baseline": baseline_summary,
        },
        "outputs": {
            "results_npz": str(out_npz),
        },
    }

    (output_dir / "tier2_collatz_run_summary_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    return manifest
