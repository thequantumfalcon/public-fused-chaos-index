from __future__ import annotations

import json
import platform
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _now_utc_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _package_version() -> str:
    try:
        from importlib.metadata import version

        return version("public-fused-chaos-index")
    except Exception:
        return "unknown"


def _python_info() -> dict[str, str]:
    return {
        "executable": sys.executable,
        "version": sys.version.replace("\n", " "),
        "platform": platform.platform(),
    }


@dataclass(frozen=True)
class StepResult:
    name: str
    status: str  # OK | SKIP | ERROR
    duration_sec: float
    details: dict[str, Any]


def _run_step(name: str, fn: Callable[[], dict[str, Any]]) -> StepResult:
    t0 = time.time()
    try:
        details = fn() or {}
        return StepResult(name=name, status="OK", duration_sec=time.time() - t0, details=details)
    except Exception as e:
        return StepResult(
            name=name,
            status="ERROR",
            duration_sec=time.time() - t0,
            details={"error": f"{type(e).__name__}: {e}"},
        )


def default_run_dir(base_output_dir: Path) -> Path:
    return base_output_dir / f"public_suite_{_now_utc_compact()}"


def run_public_suite(
    *,
    run_dir: Path,
    profile: str = "smoke",
    allow_network: bool = False,
    frontier_clusters_json: Path | None = None,
    tng_base_path: Path = Path("./TNG300-1/output"),
    skip_tng: bool = False,
    operational_n_galaxies: int = 2000,
    operational_k: int = 10,
    operational_seed: int = 42,
    operational_n_eigenstates: int = 40,
    syk_N: int = 32,
    syk_kurt_collatz: float = 3.72,
    syk_kurt_syk: float = -1.23,
    syk_method: str = "analytic",
    frontier_manifest: Path = Path("validation_results/frontier_evidence_suite_manifest.json"),
    universality_manifest: Path = Path("validation_results/universality_ground_truth_suite_manifest.json"),
) -> dict[str, Any]:
    """Run the public/offline-first suite and write a single suite manifest.

    This is intentionally lightweight:
    - No downloads
    - No bundled large artifacts
    - Produces a manifest suitable for provenance and quick sanity checks
    """

    run_dir.mkdir(parents=True, exist_ok=True)

    from .gate import check_frontier_manifest, check_universality_manifest
    from .operational import StreamlinedFCIPipeline
    from .syk_collatz import compute_syk_collatz_fci_constant

    steps: list[StepResult] = []

    prof = str(profile).strip().lower()
    if prof not in {"smoke", "offline", "full"}:
        raise ValueError("profile must be one of: smoke, offline, full")

    steps.append(
        _run_step(
            "operational",
            lambda: _run_operational(
                n_galaxies=operational_n_galaxies,
                k=operational_k,
                seed=operational_seed,
                n_eigenstates=operational_n_eigenstates,
                pipeline_cls=StreamlinedFCIPipeline,
            ),
        )
    )

    steps.append(
        _run_step(
            "syk_collatz",
            lambda: _run_syk_collatz(
                N=syk_N,
                kurt_collatz=syk_kurt_collatz,
                kurt_syk=syk_kurt_syk,
                method=syk_method,
                fn=compute_syk_collatz_fci_constant,
            ),
        )
    )

    if prof in {"offline", "full"}:
        from .suites.universality import run_universality_ground_truth_suite

        steps.append(
            _run_step(
                "universality_ground_truth",
                lambda: run_universality_ground_truth_suite(
                    output_dir=run_dir,
                    allow_network=bool(allow_network),
                    tng_base_path=tng_base_path,
                    skip_tng=bool(skip_tng),
                    k=int(operational_k),
                    n_modes=10,
                ),
            )
        )

    if prof == "full":
        if frontier_clusters_json is None:
            steps.append(
                StepResult(
                    name="frontier_evidence",
                    status="SKIP",
                    duration_sec=0.0,
                    details={"reason": "frontier_clusters_json not provided"},
                )
            )
        else:
            from .suites.frontier import run_frontier_evidence_suite

            steps.append(
                _run_step(
                    "frontier_evidence",
                    lambda: run_frontier_evidence_suite(
                        output_dir=run_dir,
                        clusters_json=frontier_clusters_json,
                        k=int(operational_k),
                        n_modes=10,
                        n_perm=2000,
                        seed=int(operational_seed),
                    ).manifest,
                )
            )

    # If this run produced suite manifests in run_dir, prefer them for gating
    # when the configured manifest paths do not exist.
    frontier_for_gate = frontier_manifest
    universality_for_gate = universality_manifest

    produced_frontier = run_dir / "frontier_evidence_suite_manifest.json"
    produced_universality = run_dir / "universality_ground_truth_suite_manifest.json"
    if not frontier_for_gate.exists() and produced_frontier.exists():
        frontier_for_gate = produced_frontier
    if not universality_for_gate.exists() and produced_universality.exists():
        universality_for_gate = produced_universality

    # Gate step is special: SKIP if manifests not present.
    gate_t0 = time.time()
    gate_details: dict[str, Any] = {
        "frontier_manifest": str(frontier_for_gate),
        "universality_manifest": str(universality_for_gate),
        "issues": [],
    }
    gate_status = "OK"

    frontier_exists = frontier_for_gate.exists()
    universality_exists = universality_for_gate.exists()
    if not frontier_exists and not universality_exists:
        gate_status = "SKIP"
        gate_details["reason"] = "no suite manifests found"
    else:
        issues: list[str] = []
        if frontier_exists:
            issues.extend(check_frontier_manifest(frontier_for_gate))
        if universality_exists:
            issues.extend(check_universality_manifest(universality_for_gate))
        gate_details["issues"] = issues
        if issues:
            gate_status = "ERROR"

    steps.append(
        StepResult(
            name="gate",
            status=gate_status,
            duration_sec=time.time() - gate_t0,
            details=gate_details,
        )
    )

    manifest: dict[str, Any] = {
        "experiment": "Public Fused Chaos Index Suite",
        "time_utc": _now_iso(),
        "package_version": _package_version(),
        "python": _python_info(),
        "run_dir": str(run_dir),
        "profile": prof,
        "allow_network": bool(allow_network),
        "steps": [
            {
                "name": s.name,
                "status": s.status,
                "duration_sec": float(s.duration_sec),
                "details": s.details,
            }
            for s in steps
        ],
        "overall_status": "OK" if all(s.status in {"OK", "SKIP"} for s in steps) else "ERROR",
    }

    (run_dir / "suite_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def _run_operational(
    *,
    n_galaxies: int,
    k: int,
    seed: int,
    n_eigenstates: int,
    pipeline_cls: type,
) -> dict[str, Any]:
    pipeline = pipeline_cls(k_neighbors=int(k), seed=int(seed))
    results = pipeline.run(n_galaxies=int(n_galaxies), n_eigenstates=int(n_eigenstates))
    fci = results["fci"]
    return {
        "n_galaxies": int(n_galaxies),
        "k": int(k),
        "seed": int(seed),
        "n_eigenstates": int(n_eigenstates),
        "fci_normalized": float(fci.fci_normalized),
        "physical_regime": str(fci.physical_regime),
    }


def _run_syk_collatz(
    *,
    N: int,
    kurt_collatz: float,
    kurt_syk: float,
    method: str,
    fn: Callable[..., Any],
) -> dict[str, Any]:
    r = fn(N=int(N), kurt_collatz=float(kurt_collatz), kurt_syk=float(kurt_syk), method=str(method))
    return {
        "N": int(N),
        "kurt_collatz": float(kurt_collatz),
        "kurt_syk": float(kurt_syk),
        "method": str(method),
        "C": float(r.C),
    }
