from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import KDTree
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigsh


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


def _positions_from_radec(ra_deg: np.ndarray, dec_deg: np.ndarray) -> np.ndarray:
    ra = np.asarray(ra_deg, dtype=np.float64).reshape(-1)
    dec = np.asarray(dec_deg, dtype=np.float64).reshape(-1)
    if ra.shape[0] != dec.shape[0]:
        raise ValueError("ra/dec length mismatch")

    ra_rad = np.radians(ra)
    dec_rad = np.radians(dec)
    x = np.cos(dec_rad) * np.cos(ra_rad)
    y = np.cos(dec_rad) * np.sin(ra_rad)
    z = np.sin(dec_rad)
    return np.column_stack([x, y, z]).astype(np.float64)


def positions_from_radec(*, ra_deg: np.ndarray, dec_deg: np.ndarray) -> np.ndarray:
    return _positions_from_radec(ra_deg, dec_deg)


def _build_knn_laplacian(positions: np.ndarray, *, k: int) -> Any:
    pos = np.asarray(positions, dtype=np.float64)
    if pos.ndim != 2 or pos.shape[0] < 3:
        raise ValueError(f"positions must be (N,D) with N>=3; got {pos.shape}")

    n = int(pos.shape[0])
    k_eff = int(min(max(int(k), 1), n - 2))

    tree = KDTree(pos)
    distances, indices = tree.query(pos, k=k_eff + 1)

    rows: list[int] = []
    cols: list[int] = []
    weights: list[float] = []

    for i in range(n):
        for j_idx in range(1, k_eff + 1):
            j = int(indices[i, j_idx])
            dist = float(distances[i, j_idx])
            w = 1.0 / (dist + 1e-10)
            rows.append(i)
            cols.append(j)
            weights.append(w)

    a = coo_matrix((weights, (rows, cols)), shape=(n, n)).tocsr()
    a = (a + a.T) * 0.5

    degrees = np.asarray(a.sum(axis=1)).reshape(-1)
    d = coo_matrix((degrees, (np.arange(n), np.arange(n))), shape=(n, n)).tocsr()
    return d - a


def compute_quantum_mass_from_positions(*, positions: np.ndarray, k: int = 10, n_modes: int = 10) -> tuple[np.ndarray, np.ndarray]:
    lap = _build_knn_laplacian(np.asarray(positions, dtype=np.float64), k=int(k))
    return _compute_quantum_mass(lap, n_modes=int(n_modes))


def _compute_quantum_mass(laplacian: Any, *, n_modes: int) -> tuple[np.ndarray, np.ndarray]:
    n = int(laplacian.shape[0])
    n_modes_eff = int(min(max(int(n_modes), 1), n - 2))

    try:
        evals, evecs = eigsh(laplacian, k=n_modes_eff, which="SA", tol=1e-6)
    except Exception:
        if n <= 2000:
            evals_all, evecs_all = np.linalg.eigh(laplacian.toarray())
            evals = evals_all[:n_modes_eff]
            evecs = evecs_all[:, :n_modes_eff]
        else:
            raise

    m = np.sum(np.asarray(evecs, dtype=np.float64) ** 2, axis=1)
    return m.astype(np.float64), np.asarray(evals, dtype=np.float64)


def add_quantum_mass_to_catalog_npz(
    *,
    catalog_npz: Path,
    out_npz: Path,
    k: int = 10,
    n_modes: int = 10,
    threshold: float = 5e-7,
    force: bool = False,
) -> dict[str, Any]:
    """Add `quantum_mass` (and `eigenvalues`) to a local NPZ catalog.

    Offline-first:
    - No network access
    - Inputs are local `.npz` artifacts

    Inputs:
    - Either `positions` (N×D) must exist, OR both `ra` and `dec` (degrees).

    Outputs:
    - Writes an output NPZ containing all original keys plus:
      - `quantum_mass` (N,)
      - `eigenvalues` (n_modes,)
      - `dark_percent` (computed from `threshold` for convenience)
    - Writes a JSON manifest alongside the output NPZ.
    """

    catalog_npz = Path(catalog_npz)
    out_npz = Path(out_npz)

    if not catalog_npz.exists():
        raise FileNotFoundError(str(catalog_npz))

    out_npz.parent.mkdir(parents=True, exist_ok=True)

    with np.load(catalog_npz, allow_pickle=True) as data:
        d = {k: data[k] for k in data.files}

    if ("quantum_mass" in d or "eigenvalues" in d) and not bool(force):
        raise ValueError("Catalog already contains quantum_mass/eigenvalues; pass force=True to overwrite")

    if "positions" in d:
        positions = np.asarray(d["positions"], dtype=np.float64)
        source = "positions"
    elif "ra" in d and "dec" in d:
        positions = _positions_from_radec(d["ra"], d["dec"])
        source = "ra_dec"
    else:
        raise KeyError("NPZ must contain `positions` or (`ra` and `dec`)")

    lap = _build_knn_laplacian(positions, k=int(k))
    qm, evals = _compute_quantum_mass(lap, n_modes=int(n_modes))

    dark_percent = float(100.0 * np.mean(qm < float(threshold))) if qm.size else float("nan")

    out_payload: dict[str, Any] = dict(d)
    out_payload["quantum_mass"] = qm
    out_payload["eigenvalues"] = evals
    out_payload["dark_percent"] = np.float64(dark_percent)

    np.savez_compressed(out_npz, **out_payload)

    manifest_path = out_npz.with_name(out_npz.stem + "_manifest.json")
    manifest = {
        "experiment": "Tier-1: Add Quantum Mass to Catalog",
        "time_utc": _now_iso(),
        "status": "OK",
        "inputs": {
            "catalog_npz": str(catalog_npz),
            "catalog_sha256": _sha256_file(catalog_npz),
            "source": source,
        },
        "params": {
            "k": int(k),
            "n_modes": int(n_modes),
            "threshold": float(threshold),
        },
        "results": {
            "n": int(qm.size),
            "quantum_mass_min": (None if qm.size == 0 else float(np.min(qm))),
            "quantum_mass_max": (None if qm.size == 0 else float(np.max(qm))),
            "quantum_mass_mean": (None if qm.size == 0 else float(np.mean(qm))),
            "dark_percent": (None if not np.isfinite(dark_percent) else float(dark_percent)),
        },
        "outputs": {
            "out_npz": str(out_npz),
            "manifest_json": str(manifest_path),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    return manifest
