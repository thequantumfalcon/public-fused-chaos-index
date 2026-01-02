from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
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


def _extract_quantum_mass(data: dict[str, np.ndarray]) -> np.ndarray:
    for key in ("quantum_mass", "mass", "M"):
        if key in data:
            return np.asarray(data[key], dtype=np.float64).reshape(-1)
    raise KeyError("input NPZ missing one of: quantum_mass, mass, M")


@dataclass(frozen=True)
class StoppingTimeResult:
    sample_size: int
    max_steps: int
    overflow_count: int
    censored_count: int
    reached_count: int
    spearman_rho_log: float
    spearman_p_log: float
    pearson_r_log: float
    pearson_p_log: float
    spearman_rho_log_ci95: tuple[float, float] | None
    pearson_r_log_ci95: tuple[float, float] | None


def _bootstrap_ci(
    x: np.ndarray,
    y: np.ndarray,
    statistic: str,
    iters: int,
    seed: int,
) -> tuple[float, float]:
    if iters <= 0:
        return (float("nan"), float("nan"))

    rng = np.random.default_rng(seed)
    n = int(x.size)
    if n < 10:
        return (float("nan"), float("nan"))

    idx = rng.integers(0, n, size=(iters, n), endpoint=False)

    vals = np.empty(iters, dtype=np.float64)
    for i in range(iters):
        xi = x[idx[i]]
        yi = y[idx[i]]
        if statistic == "spearman":
            vals[i] = spearmanr(xi, yi).statistic
        elif statistic == "pearson":
            vals[i] = pearsonr(xi, yi).statistic
        else:
            raise ValueError("statistic must be 'spearman' or 'pearson'")

    lo, hi = np.quantile(vals, [0.025, 0.975])
    return (float(lo), float(hi))


def collatz_stopping_time_uint64(n0: np.ndarray, *, max_steps: int) -> tuple[np.ndarray, np.ndarray]:
    """Compute Collatz stopping time for many starting values.

    Returns (stop_time, status):
    - stop_time: int32, 0..max_steps, max_steps means censored, -1 means overflow
    - status: int8 codes: 0=reached 1, 1=censored, 2=overflow
    """

    n = np.asarray(n0, dtype=np.uint64).copy()
    stop_time = np.zeros(n.shape, dtype=np.int32)

    status = np.zeros(n.shape, dtype=np.int8)
    active = n != np.uint64(1)

    u64_max = np.iinfo(np.uint64).max
    overflow_guard = np.uint64((u64_max - 1) // 3)

    for step in range(1, int(max_steps) + 1):
        if not np.any(active):
            break

        stop_time[active] = step

        odd = (n & np.uint64(1)) == np.uint64(1)
        odd_active = odd & active
        even_active = (~odd) & active

        if np.any(odd_active):
            will_overflow = n[odd_active] > overflow_guard
            if np.any(will_overflow):
                idx = np.flatnonzero(odd_active)
                overflow_idx = idx[will_overflow]
                status[overflow_idx] = np.int8(2)
                stop_time[overflow_idx] = np.int32(-1)
                active[overflow_idx] = False

                safe_idx = idx[~will_overflow]
                if safe_idx.size:
                    n[safe_idx] = np.uint64(3) * n[safe_idx] + np.uint64(1)
            else:
                n[odd_active] = np.uint64(3) * n[odd_active] + np.uint64(1)

        if np.any(even_active):
            n[even_active] = n[even_active] >> np.uint64(1)

        reached = (n == np.uint64(1)) & active
        if np.any(reached):
            active[reached] = False

    censored = (status == 0) & (n != np.uint64(1))
    status[censored] = np.int8(1)
    stop_time[censored] = np.int32(max_steps)

    return stop_time, status


def run_stopping_time_vs_mass(
    *,
    output_dir: Path,
    input_npz: Path,
    sample_size: int = 200_000,
    max_steps: int = 5000,
    seed: int = 42,
    bootstrap: bool = False,
    bootstrap_iters: int = 200,
) -> dict[str, Any]:
    """Tier-2 Path 3: stopping time vs quantum-mass association test.

    Offline-first: reads one local NPZ, writes results NPZ + JSON manifest.
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    input_npz = Path(input_npz)
    if not input_npz.exists():
        raise FileNotFoundError(f"Missing input NPZ: {input_npz}")

    data = np.load(input_npz, allow_pickle=True)
    mass = _extract_quantum_mass({k: data[k] for k in data.files})
    n_nodes = int(mass.size)

    rng = np.random.default_rng(int(seed))
    sample_size_eff = int(min(int(sample_size), n_nodes))

    idx = rng.integers(low=0, high=n_nodes, size=sample_size_eff, endpoint=False)
    n = (idx.astype(np.uint64) + np.uint64(1))

    stop_time, status = collatz_stopping_time_uint64(n, max_steps=int(max_steps))

    overflow_count = int(np.sum(status == 2))
    censored_count = int(np.sum(status == 1))
    reached_count = int(np.sum(status == 0))

    valid = status == 0
    if int(np.sum(valid)) < 10:
        raise RuntimeError("Too few valid samples reached 1 within horizon")

    m = mass[idx]
    st = stop_time.astype(np.float64)

    eps = np.finfo(np.float64).tiny
    log_m = np.log10(m[valid] + eps)
    log_st = np.log10(st[valid] + 1.0)

    rho_s, p_s = spearmanr(log_m, log_st)
    r_p, p_p = pearsonr(log_m, log_st)

    spearman_ci: tuple[float, float] | None = None
    pearson_ci: tuple[float, float] | None = None
    if bootstrap:
        spearman_ci = _bootstrap_ci(log_m, log_st, statistic="spearman", iters=int(bootstrap_iters), seed=int(seed))
        pearson_ci = _bootstrap_ci(log_m, log_st, statistic="pearson", iters=int(bootstrap_iters), seed=int(seed))

    res = StoppingTimeResult(
        sample_size=sample_size_eff,
        max_steps=int(max_steps),
        overflow_count=overflow_count,
        censored_count=censored_count,
        reached_count=reached_count,
        spearman_rho_log=float(rho_s),
        spearman_p_log=float(p_s),
        pearson_r_log=float(r_p),
        pearson_p_log=float(p_p),
        spearman_rho_log_ci95=spearman_ci,
        pearson_r_log_ci95=pearson_ci,
    )

    out_npz = output_dir / "tier2_path3_stopping_time_vs_mass_results.npz"
    np.savez(
        out_npz,
        input=str(input_npz),
        input_sha256=_sha256_file(input_npz),
        n_nodes=np.int64(n_nodes),
        sample_size=np.int64(sample_size_eff),
        max_steps=np.int64(max_steps),
        seed=np.int64(seed),
        overflow_count=np.int64(overflow_count),
        censored_count=np.int64(censored_count),
        reached_count=np.int64(reached_count),
        spearman_rho_log=np.float64(res.spearman_rho_log),
        spearman_p_log=np.float64(res.spearman_p_log),
        pearson_r_log=np.float64(res.pearson_r_log),
        pearson_p_log=np.float64(res.pearson_p_log),
        spearman_ci95=np.array(res.spearman_rho_log_ci95 if res.spearman_rho_log_ci95 else (np.nan, np.nan), dtype=np.float64),
        pearson_ci95=np.array(res.pearson_r_log_ci95 if res.pearson_r_log_ci95 else (np.nan, np.nan), dtype=np.float64),
    )

    manifest = {
        "experiment": "Tier-2 Path 3: Stopping Time vs Quantum Mass",
        "time_utc": _now_iso(),
        "status": "OK",
        "params": {
            "input": str(input_npz),
            "input_sha256": _sha256_file(input_npz),
            "sample_size": int(sample_size_eff),
            "max_steps": int(max_steps),
            "seed": int(seed),
            "bootstrap": bool(bootstrap),
            "bootstrap_iters": int(bootstrap_iters),
        },
        "results": {
            "n_nodes": int(n_nodes),
            "overflow_count": int(overflow_count),
            "censored_count": int(censored_count),
            "reached_count": int(reached_count),
            "spearman_rho_log": float(res.spearman_rho_log),
            "spearman_p_log": float(res.spearman_p_log),
            "pearson_r_log": float(res.pearson_r_log),
            "pearson_p_log": float(res.pearson_p_log),
            "spearman_rho_log_ci95": res.spearman_rho_log_ci95,
            "pearson_r_log_ci95": res.pearson_r_log_ci95,
        },
        "outputs": {
            "results_npz": str(out_npz),
        },
    }

    (output_dir / "tier2_path3_stopping_time_vs_mass_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    return manifest
