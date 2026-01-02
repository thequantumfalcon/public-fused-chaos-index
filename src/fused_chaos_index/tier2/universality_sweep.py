from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

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


def _entropy_bits_from_hist(values: np.ndarray, *, bins: int = 80) -> float:
    if values.size == 0:
        return float("nan")
    hist, _ = np.histogram(values, bins=bins)
    p = hist.astype(np.float64)
    p_sum = p.sum()
    if p_sum == 0:
        return float("nan")
    p /= p_sum
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def _safe_log10(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    eps = np.finfo(np.float64).tiny
    return np.log10(x + eps)


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def _extract_quantum_mass(data: dict[str, np.ndarray]) -> np.ndarray:
    for key in ("quantum_mass", "mass", "M"):
        if key in data:
            return np.asarray(data[key], dtype=np.float64).reshape(-1)
    raise KeyError("input NPZ missing one of: quantum_mass, mass, M")


@dataclass(frozen=True)
class RunSummary:
    file: str
    sha256: str
    n: int
    dtype: str
    k: int | None
    dark_percent_file: float | None
    dark_percent_common: float
    mass_min: float
    mass_p10: float
    mass_median: float
    mass_p90: float
    mass_mean: float
    entropy_bits: float


def run_universality_sweep(
    *,
    output_dir: Path,
    inputs: Iterable[Path],
    threshold: float = 5e-7,
    plot: bool = False,
) -> dict[str, Any]:
    """Tier-2 Path 1: compare multiple quantum-mass artifacts under a shared threshold.

    This is offline-first: it only reads local NPZ inputs and writes local outputs.

    Each input NPZ must contain one of the keys:
    - quantum_mass (preferred)
    - mass
    - M

    Optional keys used if present:
    - dark_percent (original run's reported percentage)
    - k_modes or eigenvalues (used to infer k)
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    in_paths = [Path(p) for p in inputs]
    if not in_paths:
        raise ValueError("inputs must be a non-empty list of paths")

    summaries: list[RunSummary] = []
    issues: list[str] = []

    for p in in_paths:
        if not p.exists():
            issues.append(f"missing input: {p}")
            continue

        try:
            data = _load_npz(p)
            mass = _extract_quantum_mass(data)
            n = int(mass.size)

            k_val: int | None
            if "k_modes" in data:
                try:
                    k_val = int(np.asarray(data["k_modes"]))
                except Exception:
                    k_val = None
            elif "eigenvalues" in data:
                try:
                    k_val = int(np.asarray(data["eigenvalues"]).size)
                except Exception:
                    k_val = None
            else:
                k_val = None

            dark_percent_file: float | None = None
            if "dark_percent" in data:
                try:
                    dark_percent_file = float(np.asarray(data["dark_percent"]))
                except Exception:
                    dark_percent_file = None

            dark_percent_common = float(100.0 * np.mean(mass < float(threshold)))

            logm = _safe_log10(mass)
            summary = RunSummary(
                file=str(p.name),
                sha256=_sha256_file(p),
                n=n,
                dtype=str(mass.dtype),
                k=k_val,
                dark_percent_file=dark_percent_file,
                dark_percent_common=dark_percent_common,
                mass_min=float(np.min(mass)) if n else float("nan"),
                mass_p10=float(np.quantile(mass, 0.10)) if n else float("nan"),
                mass_median=float(np.quantile(mass, 0.50)) if n else float("nan"),
                mass_p90=float(np.quantile(mass, 0.90)) if n else float("nan"),
                mass_mean=float(np.mean(mass)) if n else float("nan"),
                entropy_bits=_entropy_bits_from_hist(logm, bins=80),
            )
            summaries.append(summary)
        except Exception as e:
            issues.append(f"{p}: {type(e).__name__}: {e}")

    summaries.sort(key=lambda r: r.n)

    out_npz = output_dir / "tier2_path1_universality_results.npz"
    np.savez(
        out_npz,
        threshold=np.float64(threshold),
        input_file=np.array([r.file for r in summaries]),
        input_sha256=np.array([r.sha256 for r in summaries]),
        n=np.array([r.n for r in summaries], dtype=np.int64),
        k=np.array([-1 if r.k is None else r.k for r in summaries], dtype=np.int64),
        dtype=np.array([r.dtype for r in summaries]),
        dark_percent_file=np.array(
            [np.nan if r.dark_percent_file is None else r.dark_percent_file for r in summaries],
            dtype=np.float64,
        ),
        dark_percent_common=np.array([r.dark_percent_common for r in summaries], dtype=np.float64),
        mass_min=np.array([r.mass_min for r in summaries], dtype=np.float64),
        mass_p10=np.array([r.mass_p10 for r in summaries], dtype=np.float64),
        mass_median=np.array([r.mass_median for r in summaries], dtype=np.float64),
        mass_p90=np.array([r.mass_p90 for r in summaries], dtype=np.float64),
        mass_mean=np.array([r.mass_mean for r in summaries], dtype=np.float64),
        entropy_bits=np.array([r.entropy_bits for r in summaries], dtype=np.float64),
    )

    plot_path: str | None = None
    if plot:
        try:
            import matplotlib.pyplot as plt

            n_arr = np.array([r.n for r in summaries], dtype=np.float64)
            dp = np.array([r.dark_percent_common for r in summaries], dtype=np.float64)

            fig, ax = plt.subplots(figsize=(8, 4.5))
            ax.plot(n_arr, dp, marker="o")
            ax.set_xscale("log")
            ax.set_xlabel("N (nodes)")
            ax.set_ylabel("dark% under shared threshold")
            ax.grid(True, alpha=0.3)

            plot_path_obj = output_dir / "tier2_path1_universality.png"
            fig.tight_layout()
            fig.savefig(plot_path_obj, dpi=150)
            plt.close(fig)
            plot_path = str(plot_path_obj)
        except Exception as e:
            issues.append(f"plot: {type(e).__name__}: {e}")

    manifest = {
        "experiment": "Tier-2 Path 1: Ergodic Universality Sweep",
        "time_utc": _now_iso(),
        "status": "OK" if summaries and not issues else ("ERROR" if issues else "SKIP"),
        "params": {
            "threshold": float(threshold),
            "plot": bool(plot),
        },
        "inputs": [asdict(r) for r in summaries],
        "outputs": {
            "results_npz": str(out_npz),
            "plot_png": plot_path,
        },
        "issues": issues,
    }

    (output_dir / "tier2_path1_universality_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    return manifest
