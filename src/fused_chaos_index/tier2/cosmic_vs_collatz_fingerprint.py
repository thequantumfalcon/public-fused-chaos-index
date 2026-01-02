from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import pearsonr, spearmanr


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


def _entropy_bits(values: np.ndarray, *, bins: int = 80) -> float:
    if values.size == 0:
        return float("nan")
    hist, _ = np.histogram(values, bins=bins)
    p = hist.astype(np.float64)
    s = p.sum()
    if s == 0:
        return float("nan")
    p /= s
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def _safe_log10(x: np.ndarray) -> np.ndarray:
    eps = np.finfo(np.float64).tiny
    return np.log10(np.asarray(x, dtype=np.float64) + eps)


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


@dataclass(frozen=True)
class Fingerprint:
    name: str
    n: int
    log10_q05: float
    log10_q25: float
    log10_q50: float
    log10_q75: float
    log10_q95: float
    mean: float
    median: float
    frac_below_threshold: float
    entropy_bits: float


def _fingerprint(name: str, values: np.ndarray, *, threshold: float) -> Fingerprint:
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    lx = _safe_log10(x)

    q05, q25, q50, q75, q95 = np.quantile(lx, [0.05, 0.25, 0.50, 0.75, 0.95])

    return Fingerprint(
        name=str(name),
        n=int(x.size),
        log10_q05=float(q05),
        log10_q25=float(q25),
        log10_q50=float(q50),
        log10_q75=float(q75),
        log10_q95=float(q95),
        mean=float(np.mean(x)) if x.size else float("nan"),
        median=float(np.median(x)) if x.size else float("nan"),
        frac_below_threshold=float(np.mean(x < float(threshold))) if x.size else float("nan"),
        entropy_bits=_entropy_bits(lx, bins=80),
    )


def run_cosmic_vs_collatz_fingerprint(
    *,
    output_dir: Path,
    collatz_npz: Path,
    smacs_npz: Path,
    flamingo_npz: Path | None = None,
    threshold: float = 5e-7,
) -> dict[str, Any]:
    """Tier-2 Path 2: cross-domain fingerprint comparison.

    Inputs are purely local artifacts (NPZ files). This runner computes:
    - distribution fingerprints of `quantum_mass` and (for SMACS) `kappa`
    - correlations between SMACS quantum mass and kappa
    - optional Collatz low-energy spectrum summaries (if eigenvalues included)

    It does not claim physical causality; it only writes descriptive statistics.
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    collatz_npz = Path(collatz_npz)
    smacs_npz = Path(smacs_npz)
    flamingo_npz = Path(flamingo_npz) if flamingo_npz is not None else None

    if not collatz_npz.exists():
        raise FileNotFoundError(f"Missing Collatz input: {collatz_npz}")
    if not smacs_npz.exists():
        raise FileNotFoundError(f"Missing SMACS input: {smacs_npz}")

    collatz = _load_npz(collatz_npz)
    if "quantum_mass" not in collatz:
        raise KeyError("Collatz NPZ missing key: quantum_mass")
    collatz_mass = np.asarray(collatz["quantum_mass"], dtype=np.float64)

    smacs = _load_npz(smacs_npz)
    if "quantum_mass" not in smacs or "kappa" not in smacs:
        raise KeyError("SMACS NPZ must contain keys: quantum_mass, kappa")
    smacs_mass = np.asarray(smacs["quantum_mass"], dtype=np.float64)
    smacs_kappa = np.asarray(smacs["kappa"], dtype=np.float64)

    fp_collatz = _fingerprint("Collatz", collatz_mass, threshold=float(threshold))
    fp_smacs_mass = _fingerprint("SMACS_M", smacs_mass, threshold=float(threshold))
    fp_smacs_kappa = _fingerprint("SMACS_kappa", smacs_kappa, threshold=float(threshold))

    rho_s, p_s = spearmanr(smacs_mass, smacs_kappa)
    r_p, p_p = pearsonr(smacs_mass, smacs_kappa)

    evals_sorted = np.array([], dtype=np.float64)
    first_gap = float("nan")
    gap_ratio = float("nan")
    if "eigenvalues" in collatz:
        try:
            evals = np.asarray(collatz["eigenvalues"], dtype=np.float64).reshape(-1)
            evals_sorted = np.sort(evals)
            if evals_sorted.size >= 2:
                first_gap = float(evals_sorted[1] - evals_sorted[0])
                if evals_sorted[0] != 0.0:
                    gap_ratio = float(first_gap / evals_sorted[0])
        except Exception:
            evals_sorted = np.array([], dtype=np.float64)

    flamingo_loaded = False
    flamingo_keys: list[str] = []
    fp_flamingo_mass: Fingerprint | None = None
    flamingo_rho: float | None = None
    flamingo_p_value: float | None = None

    if flamingo_npz is not None and flamingo_npz.exists():
        flamingo = _load_npz(flamingo_npz)
        flamingo_loaded = True
        flamingo_keys = sorted(list(flamingo.keys()))

        if "mass" in flamingo:
            fp_flamingo_mass = _fingerprint("FLAMINGO_mass", np.asarray(flamingo["mass"], dtype=np.float64), threshold=float(threshold))
        if "rho" in flamingo:
            try:
                flamingo_rho = float(np.asarray(flamingo["rho"]))
            except Exception:
                flamingo_rho = None
        if "p_value" in flamingo:
            try:
                flamingo_p_value = float(np.asarray(flamingo["p_value"]))
            except Exception:
                flamingo_p_value = None

    out_npz = output_dir / "tier2_path2_fingerprint_results.npz"
    np.savez(
        out_npz,
        threshold=np.float64(threshold),
        collatz_file=str(collatz_npz),
        collatz_sha256=_sha256_file(collatz_npz),
        smacs_file=str(smacs_npz),
        smacs_sha256=_sha256_file(smacs_npz),
        flamingo_file=(str(flamingo_npz) if flamingo_npz is not None else ""),
        flamingo_sha256=(_sha256_file(flamingo_npz) if flamingo_npz is not None and flamingo_npz.exists() else ""),
        collatz_fingerprint=np.array([asdict(fp_collatz)], dtype=object),
        smacs_mass_fingerprint=np.array([asdict(fp_smacs_mass)], dtype=object),
        smacs_kappa_fingerprint=np.array([asdict(fp_smacs_kappa)], dtype=object),
        smacs_spearman_rho=np.float64(rho_s),
        smacs_spearman_p=np.float64(p_s),
        smacs_pearson_r=np.float64(r_p),
        smacs_pearson_p=np.float64(p_p),
        collatz_eigenvalues=evals_sorted,
        collatz_first_gap=np.float64(first_gap),
        collatz_gap_ratio=np.float64(gap_ratio),
        flamingo_loaded=np.bool_(flamingo_loaded),
        flamingo_keys=np.array(flamingo_keys, dtype=object),
        flamingo_mass_fingerprint=(
            np.array([asdict(fp_flamingo_mass)], dtype=object) if fp_flamingo_mass is not None else np.array([], dtype=object)
        ),
        flamingo_rho=(np.float64(flamingo_rho) if flamingo_rho is not None else np.float64(np.nan)),
        flamingo_p_value=(np.float64(flamingo_p_value) if flamingo_p_value is not None else np.float64(np.nan)),
    )

    manifest = {
        "experiment": "Tier-2 Path 2: Cosmic vs Collatz Fingerprint",
        "time_utc": _now_iso(),
        "status": "OK",
        "params": {
            "threshold": float(threshold),
            "collatz": str(collatz_npz),
            "collatz_sha256": _sha256_file(collatz_npz),
            "smacs": str(smacs_npz),
            "smacs_sha256": _sha256_file(smacs_npz),
            "flamingo": (str(flamingo_npz) if flamingo_npz is not None else None),
            "flamingo_sha256": (_sha256_file(flamingo_npz) if flamingo_npz is not None and flamingo_npz.exists() else None),
        },
        "fingerprints": {
            "collatz": asdict(fp_collatz),
            "smacs_mass": asdict(fp_smacs_mass),
            "smacs_kappa": asdict(fp_smacs_kappa),
            "flamingo_mass": (asdict(fp_flamingo_mass) if fp_flamingo_mass is not None else None),
        },
        "stats": {
            "smacs_spearman_rho": float(rho_s),
            "smacs_spearman_p": float(p_s),
            "smacs_pearson_r": float(r_p),
            "smacs_pearson_p": float(p_p),
            "collatz_spectrum": {
                "k": int(evals_sorted.size),
                "min": (float(evals_sorted[0]) if evals_sorted.size else None),
                "max": (float(evals_sorted[-1]) if evals_sorted.size else None),
                "first_gap": (float(first_gap) if evals_sorted.size >= 2 else None),
                "gap_ratio": (float(gap_ratio) if evals_sorted.size >= 2 else None),
            },
            "flamingo": {
                "loaded": bool(flamingo_loaded),
                "keys": flamingo_keys,
                "stored_rho": (float(flamingo_rho) if flamingo_rho is not None else None),
                "stored_p_value": (float(flamingo_p_value) if flamingo_p_value is not None else None),
            },
        },
        "outputs": {
            "results_npz": str(out_npz),
        },
    }

    (output_dir / "tier2_path2_fingerprint_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    return manifest
