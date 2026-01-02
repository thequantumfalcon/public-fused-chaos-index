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
class BolshoiResult:
    status: str  # OK | SKIP | ERROR
    results: dict[str, Any]


def load_bolshoi_halos() -> tuple[np.ndarray, np.ndarray]:
    """Return (positions, halo_mvir) from the Halotools Bolshoi cache.

    This is optional functionality. If `halotools`/`h5py` are missing, callers
    should treat it as SKIP.
    """

    import h5py  # type: ignore
    from halotools.custom_exceptions import HalotoolsError  # type: ignore
    from halotools.sim_manager import DownloadManager, HaloTableCache, sim_defaults  # type: ignore

    cache = HaloTableCache()
    cache.update_log_from_current_ascii()

    def _find_cached_entry() -> Any | None:
        matches = list(
            cache.matching_log_entry_generator(
                simname="bolshoi",
                halo_finder="rockstar",
                version_name=sim_defaults.default_version_name,
                redshift=0.0,
                dz_tol=0.1,
            )
        )
        return matches[0] if matches else None

    entry = _find_cached_entry()
    if entry is None:
        dman = DownloadManager()
        try:
            dman.download_processed_halo_table(
                simname="bolshoi",
                halo_finder="rockstar",
                version_name=sim_defaults.default_version_name,
                redshift=0.0,
            )
        except HalotoolsError as exc:
            raise RuntimeError(f"Halotools download failed: {exc}") from exc

        cache.update_log_from_current_ascii()
        entry = _find_cached_entry()

    if entry is None:
        raise RuntimeError("Unable to locate Bolshoi catalog in Halotools cache")

    fname = str(getattr(entry, "fname"))
    with h5py.File(fname, "r") as f:
        data: Any = f["data"]
        x = np.asarray(data["halo_x"], dtype=np.float64)
        y = np.asarray(data["halo_y"], dtype=np.float64)
        z = np.asarray(data["halo_z"], dtype=np.float64)
        positions = np.vstack([x, y, z]).T
        halo_mvir = np.asarray(data["halo_mvir"], dtype=np.float64)

    return positions, halo_mvir


def choose_sample(
    positions: np.ndarray,
    halo_mvir: np.ndarray,
    max_n: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if max_n <= 0 or max_n >= len(positions):
        return positions, halo_mvir
    rng = np.random.default_rng(int(seed))
    idx = rng.choice(len(positions), size=int(max_n), replace=False)
    return positions[idx], halo_mvir[idx]


def build_knn_hamiltonian(positions: np.ndarray, k: int) -> Any:
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


def compute_quantum_mass(h: Any, n_modes: int) -> tuple[np.ndarray, np.ndarray]:
    evals, evecs = eigsh(h, k=int(n_modes), which="SA")
    m = np.sum(evecs[:, : int(n_modes)] ** 2, axis=1)
    return m.astype(np.float64), evals.astype(np.float64)


def analyze(M: np.ndarray, mvir: np.ndarray) -> dict[str, Any]:
    rho_s, p_s = _spearman(M, mvir)
    r_p, p_p = _pearson(M, mvir)

    log_M = np.log10(M + 1e-12)
    log_mvir = np.log10(mvir + 1e-12)
    rho_s_log, p_s_log = _spearman(log_M, log_mvir)
    r_p_log, p_p_log = _pearson(log_M, log_mvir)

    verdict = "FAIL"
    if abs(rho_s) > 0.3 and p_s < 1e-6:
        verdict = "PASS"
    elif abs(rho_s) > 0.2 and p_s < 1e-3:
        verdict = "MODERATE"

    return {
        "n": int(len(M)),
        "rho_spearman": float(rho_s),
        "p_spearman": float(p_s),
        "r_pearson": float(r_p),
        "p_pearson": float(p_p),
        "rho_spearman_log": float(rho_s_log),
        "p_spearman_log": float(p_s_log),
        "r_pearson_log": float(r_p_log),
        "p_pearson_log": float(p_p_log),
        "verdict": verdict,
    }


def run_bolshoi_ground_truth(
    *,
    output_dir: Path,
    max_n: int = 5000,
    k: int = 10,
    n_modes: int = 10,
    seed: int = 42,
    allow_network: bool = False,
) -> BolshoiResult:
    """Run the Bolshoi (Halotools) ground-truth validator.

    Data-safety / low-friction:
    - If optional deps are missing, returns SKIP.
    - If download would be required and allow_network is False, returns SKIP.
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "halotools_bolshoi_ground_truth_manifest.json"

    try:
        import halotools  # noqa: F401
        import h5py  # noqa: F401
    except Exception as e:
        return BolshoiResult(status="SKIP", results={"reason": f"missing optional deps: {type(e).__name__}: {e}"})

    # Prevent implicit downloads unless explicitly allowed.
    if not allow_network:
        try:
            from halotools.sim_manager import HaloTableCache, sim_defaults  # type: ignore

            cache = HaloTableCache()
            cache.update_log_from_current_ascii()
            matches = list(
                cache.matching_log_entry_generator(
                    simname="bolshoi",
                    halo_finder="rockstar",
                    version_name=sim_defaults.default_version_name,
                    redshift=0.0,
                    dz_tol=0.1,
                )
            )
            if not matches:
                return BolshoiResult(
                    status="SKIP",
                    results={"reason": "Bolshoi not in Halotools cache and allow_network is False"},
                )
        except Exception:
            # If the cache check fails, still attempt; load_bolshoi_halos will error if missing.
            pass

    try:
        positions, mvir = load_bolshoi_halos()
        positions, mvir = choose_sample(positions, mvir, max_n=int(max_n), seed=int(seed))
        h = build_knn_hamiltonian(positions, k=int(k))
        M, evals = compute_quantum_mass(h, n_modes=int(n_modes))
        res = analyze(M, mvir)
        payload = {
            "experiment": "Bolshoi (Halotools) Ground-Truth Validation",
            "time_utc": _now_iso(),
            "status": "OK",
            "params": {"max_n": int(max_n), "k": int(k), "n_modes": int(n_modes), "seed": int(seed)},
            "results": {**res, "eigenvalues": evals.tolist()},
        }
        manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return BolshoiResult(status="OK", results=payload)
    except Exception as e:
        payload = {
            "experiment": "Bolshoi (Halotools) Ground-Truth Validation",
            "time_utc": _now_iso(),
            "status": "ERROR",
            "error": f"{type(e).__name__}: {e}",
        }
        manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return BolshoiResult(status="ERROR", results=payload)
