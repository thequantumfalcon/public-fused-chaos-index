from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import KDTree
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigsh
from scipy.stats import spearmanr


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _as_float(x: Any) -> float:
    return float(np.asarray(x).reshape(-1)[0])


def _spearman(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    res: Any = spearmanr(x, y)
    if hasattr(res, "correlation") and hasattr(res, "pvalue"):
        return _as_float(res.correlation), _as_float(res.pvalue)
    if hasattr(res, "statistic") and hasattr(res, "pvalue"):
        return _as_float(res.statistic), _as_float(res.pvalue)
    rho, p = res
    return _as_float(rho), _as_float(p)


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


def _permutation_p_value(
    *,
    M: np.ndarray,
    y: np.ndarray,
    rho_obs: float,
    n_perm: int,
    rng: np.random.Generator,
) -> float:
    if int(n_perm) <= 0:
        return float("nan")

    more_extreme = 0
    for _ in range(int(n_perm)):
        yp = rng.permutation(y)
        rho_p, _ = _spearman(M, yp)
        if abs(rho_p) >= abs(float(rho_obs)):
            more_extreme += 1

    # +1 smoothing avoids 0 and 1 exactly for finite samples.
    return float((more_extreme + 1) / (int(n_perm) + 1))


@dataclass(frozen=True)
class FrontierSuiteResult:
    status: str  # OK | SKIP | ERROR
    manifest: dict[str, Any]


def run_frontier_evidence_suite(
    *,
    output_dir: Path,
    clusters_json: Path | None = None,
    k: int = 10,
    n_modes: int = 10,
    n_perm: int = 2000,
    seed: int = 42,
) -> FrontierSuiteResult:
    """Run an offline-first Frontier-style evidence suite over local artifacts.

    Input format
    - clusters_json: JSON file with a top-level list `clusters`, each containing:
        - cluster_id: str/int
        - cluster_name: optional str
        - npz_path: path to a local .npz containing arrays:
            - positions: (N, D)
            - kappa: (N,)

    Safety
    - No downloads.
    - Returns SKIP if clusters_json is not provided or doesn't exist.
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "frontier_evidence_suite_manifest.json"

    if clusters_json is None or not clusters_json.exists():
        payload = {
            "experiment": "Frontier Evidence Suite",
            "time_utc": _now_iso(),
            "status": "SKIP",
            "reason": "clusters_json not provided or not found",
            "clusters": [],
            "params": {"k": int(k), "n_modes": int(n_modes), "n_perm": int(n_perm), "seed": int(seed)},
        }
        manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return FrontierSuiteResult(status="SKIP", manifest=payload)

    try:
        spec = json.loads(clusters_json.read_text(encoding="utf-8"))
        clusters = spec.get("clusters")
        if not isinstance(clusters, list) or not clusters:
            raise ValueError("clusters_json missing or empty 'clusters' list")

        rng = np.random.default_rng(int(seed))
        out_clusters: list[dict[str, Any]] = []

        for c in clusters:
            cid = str(c.get("cluster_id", "?"))
            cname = str(c.get("cluster_name", cid))
            npz_path = Path(str(c.get("npz_path")))
            if not npz_path.is_absolute():
                npz_path = (clusters_json.parent / npz_path).resolve()

            if not npz_path.exists():
                out_clusters.append(
                    {
                        "cluster_id": cid,
                        "cluster_name": cname,
                        "status": "ERROR",
                        "error": f"npz_path not found: {npz_path}",
                    }
                )
                continue

            sha = _sha256_file(npz_path)
            data = np.load(npz_path)
            positions = np.asarray(data["positions"], dtype=np.float64)
            kappa = np.asarray(data["kappa"], dtype=np.float64).reshape(-1)

            if positions.shape[0] != kappa.shape[0]:
                raise ValueError(f"cluster {cid}: positions and kappa length mismatch")
            if positions.ndim != 2:
                raise ValueError(f"cluster {cid}: positions must be 2D")
            if positions.shape[0] < max(int(k) + 2, int(n_modes) + 2):
                raise ValueError(f"cluster {cid}: not enough points for k/n_modes")

            h = _build_knn_hamiltonian(positions, k=int(k))
            M, evals = _compute_quantum_mass(h, n_modes=int(n_modes))

            rho, p = _spearman(M, kappa)
            p_perm = _permutation_p_value(M=M, y=kappa, rho_obs=rho, n_perm=int(n_perm), rng=rng)

            out_clusters.append(
                {
                    "cluster_id": cid,
                    "cluster_name": cname,
                    "status": "OK",
                    "correlation": {
                        "statistic": "spearman",
                        "spearman_rho": float(rho),
                        "spearman_p": float(p),
                    },
                    "null_test": {
                        "method": "permutation",
                        "n_perm": int(n_perm),
                        "p_permutation": float(p_perm),
                    },
                    "provenance": {
                        "npz_path": str(npz_path),
                        "kappa_sha256": str(sha),
                        "fields": ["positions", "kappa"],
                    },
                    "notes": {
                        "n": int(len(M)),
                        "k": int(k),
                        "n_modes": int(n_modes),
                        "eigenvalues": evals.tolist(),
                    },
                }
            )

        payload = {
            "experiment": "Frontier Evidence Suite",
            "time_utc": _now_iso(),
            "status": "OK" if all(c.get("status") == "OK" for c in out_clusters) else "ERROR",
            "clusters": out_clusters,
            "params": {
                "clusters_json": str(clusters_json),
                "k": int(k),
                "n_modes": int(n_modes),
                "n_perm": int(n_perm),
                "seed": int(seed),
            },
        }
        manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return FrontierSuiteResult(status=str(payload["status"]), manifest=payload)
    except Exception as e:
        payload = {
            "experiment": "Frontier Evidence Suite",
            "time_utc": _now_iso(),
            "status": "ERROR",
            "error": f"{type(e).__name__}: {e}",
        }
        manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return FrontierSuiteResult(status="ERROR", manifest=payload)
