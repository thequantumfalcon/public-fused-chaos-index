from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Tuple

import numpy as np
from scipy.spatial import KDTree
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigsh
from scipy.stats import pearsonr, spearmanr


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _as_float(x: Any) -> float:
    return float(np.asarray(x).reshape(-1)[0])


def _spearman(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    res: Any = spearmanr(x, y)
    if hasattr(res, "correlation") and hasattr(res, "pvalue"):
        return _as_float(res.correlation), _as_float(res.pvalue)
    if hasattr(res, "statistic") and hasattr(res, "pvalue"):
        return _as_float(res.statistic), _as_float(res.pvalue)
    rho, p = res
    return _as_float(rho), _as_float(p)


def _pearson(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    res: Any = pearsonr(x, y)
    if hasattr(res, "correlation") and hasattr(res, "pvalue"):
        return _as_float(res.correlation), _as_float(res.pvalue)
    if hasattr(res, "statistic") and hasattr(res, "pvalue"):
        return _as_float(res.statistic), _as_float(res.pvalue)
    r, p = res
    return _as_float(r), _as_float(p)


@dataclass(frozen=True)
class TNGResult:
    status: str  # OK | SKIP | ERROR
    results: dict[str, Any]


def _build_knn_hamiltonian(positions: np.ndarray, k: int) -> Any:
    n = positions.shape[0]
    tree = KDTree(positions)
    distances, indices = tree.query(positions, k=int(k) + 1)

    rows: list[int] = []
    cols: list[int] = []
    weights: list[float] = []

    for i in range(n):
        for j_idx in range(1, int(k) + 1):
            j = int(indices[i, j_idx])
            dist = float(distances[i, j_idx])
            w = 1.0 / (1.0 + dist)
            rows.append(i)
            cols.append(j)
            weights.append(w)

    h = coo_matrix((weights, (rows, cols)), shape=(n, n)).tocsr()
    return (h + h.T) * 0.5


def _compute_quantum_mass(h: Any, n_modes: int) -> tuple[np.ndarray, np.ndarray]:
    evals, evecs = eigsh(h, k=int(n_modes), which="SA")
    m = np.sum(evecs[:, : int(n_modes)] ** 2, axis=1)
    return m.astype(np.float64), evals.astype(np.float64)


def run_tng_ground_truth(
    *,
    base_path: Path,
    output_dir: Path,
    snapshot: int = 99,
    min_stellar_mass: float = 1e9,
    max_n: int = 5000,
    k: int = 10,
    n_modes: int = 10,
    seed: int = 42,
) -> TNGResult:
    """Run a lightweight IllustrisTNG ground-truth validator.

    Safe behavior:
    - SKIP if `illustris_python` isn't available.
    - SKIP if base_path doesn't exist or doesn't contain the needed group catalog.
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "tng_validation_manifest.json"

    try:
        import importlib.util

        if importlib.util.find_spec("illustris_python") is None:
            return TNGResult(status="SKIP", results={"reason": "missing optional dep: illustris_python"})
        import illustris_python as il  # type: ignore
    except Exception as e:
        return TNGResult(status="SKIP", results={"reason": f"illustris_python unavailable: {type(e).__name__}: {e}"})

    if not base_path.exists():
        return TNGResult(status="SKIP", results={"reason": f"base_path not found: {base_path}"})

    try:
        fields = ["SubhaloPos", "SubhaloMassType"]
        sub = il.groupcat.loadSubhalos(str(base_path), int(snapshot), fields=fields)
        positions = np.asarray(sub["SubhaloPos"], dtype=np.float64)
        mass_types = np.asarray(sub["SubhaloMassType"], dtype=np.float64)

        h = 0.6774
        dm_mass_true = mass_types[:, 1] * 1e10 / h
        stellar_mass = mass_types[:, 4] * 1e10 / h

        mask = stellar_mass > float(min_stellar_mass)
        positions = positions[mask]
        dm_mass_true = dm_mass_true[mask]

        # Subsample for tractability.
        if int(max_n) > 0 and positions.shape[0] > int(max_n):
            rng = np.random.default_rng(int(seed))
            idx = rng.choice(positions.shape[0], size=int(max_n), replace=False)
            positions = positions[idx]
            dm_mass_true = dm_mass_true[idx]

        hmat = _build_knn_hamiltonian(positions, k=int(k))
        M, evals = _compute_quantum_mass(hmat, n_modes=int(n_modes))

        rho_s, p_s = _spearman(M, dm_mass_true)
        r_p, p_p = _pearson(M, dm_mass_true)
        rho_s_log, p_s_log = _spearman(np.log10(M + 1e-12), np.log10(dm_mass_true + 1e-12))

        verdict = "FAIL"
        if abs(rho_s) > 0.3 and p_s < 1e-6:
            verdict = "PASS"
        elif abs(rho_s) > 0.2 and p_s < 1e-3:
            verdict = "MODERATE"

        payload = {
            "experiment": "IllustrisTNG Ground-Truth Validation",
            "time_utc": _now_iso(),
            "status": "OK",
            "params": {
                "snapshot": int(snapshot),
                "min_stellar_mass": float(min_stellar_mass),
                "max_n": int(max_n),
                "k": int(k),
                "n_modes": int(n_modes),
                "seed": int(seed),
            },
            "results": {
                "n": int(len(M)),
                "rho_spearman": float(rho_s),
                "p_spearman": float(p_s),
                "r_pearson": float(r_p),
                "p_pearson": float(p_p),
                "rho_spearman_log": float(rho_s_log),
                "p_spearman_log": float(p_s_log),
                "verdict": verdict,
                "eigenvalues": evals.tolist(),
            },
        }
        manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return TNGResult(status="OK", results=payload)
    except Exception as e:
        payload = {
            "experiment": "IllustrisTNG Ground-Truth Validation",
            "time_utc": _now_iso(),
            "status": "ERROR",
            "error": f"{type(e).__name__}: {e}",
        }
        manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return TNGResult(status="ERROR", results=payload)
