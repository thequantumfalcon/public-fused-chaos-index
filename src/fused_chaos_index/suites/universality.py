from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from ..validators.bolshoi import run_bolshoi_ground_truth
from ..validators.tng import run_tng_ground_truth


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass(frozen=True)
class RunResult:
    dataset: str
    status: str  # OK | SKIP | ERROR
    details: Dict[str, Any]


def _extract_row(result: RunResult) -> dict[str, Any]:
    if result.status != "OK":
        return {
            "dataset": result.dataset,
            "status": result.status,
            "n": "",
            "rho_spearman": "",
            "p_spearman": "",
            "rho_spearman_log": "",
            "p_spearman_log": "",
            "verdict": "",
            "note": result.details.get("reason") or result.details.get("error") or "",
        }

    res = result.details.get("results", {})
    return {
        "dataset": result.dataset,
        "status": result.status,
        "n": res.get("n") or res.get("n_galaxies") or "",
        "rho_spearman": res.get("rho_spearman", ""),
        "p_spearman": res.get("p_spearman", ""),
        "rho_spearman_log": res.get("rho_spearman_log", ""),
        "p_spearman_log": res.get("p_spearman_log", ""),
        "verdict": res.get("verdict", ""),
        "note": "",
    }


def run_universality_ground_truth_suite(
    *,
    output_dir: Path,
    allow_network: bool = False,
    tng_base_path: Path = Path("./TNG300-1/output"),
    k: int = 10,
    n_modes: int = 10,
    bolshoi_max_n: int = 5000,
    skip_tng: bool = False,
) -> dict[str, Any]:
    """Run TNG + Bolshoi validators and write a suite manifest + comparison CSV."""

    output_dir.mkdir(parents=True, exist_ok=True)

    if skip_tng:
        tng_res = RunResult(dataset="IllustrisTNG", status="SKIP", details={"reason": "--skip-tng"})
    else:
        tng = run_tng_ground_truth(base_path=tng_base_path, output_dir=output_dir, k=k, n_modes=n_modes)
        tng_res = RunResult(dataset="IllustrisTNG", status=tng.status, details=tng.results)

    bol = run_bolshoi_ground_truth(
        output_dir=output_dir,
        max_n=bolshoi_max_n,
        k=k,
        n_modes=n_modes,
        allow_network=allow_network,
    )
    bol_res = RunResult(dataset="Bolshoi (Halotools)", status=bol.status, details=bol.results)

    rows = [_extract_row(tng_res), _extract_row(bol_res)]
    csv_path = output_dir / "universality_ground_truth_comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dataset",
                "status",
                "n",
                "rho_spearman",
                "p_spearman",
                "rho_spearman_log",
                "p_spearman_log",
                "verdict",
                "note",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    suite_manifest = {
        "experiment": "Universality Ground-Truth Suite",
        "time_utc": _now_iso(),
        "results": {
            "tng": {"status": tng_res.status, **tng_res.details},
            "bolshoi": {"status": bol_res.status, **bol_res.details},
        },
        "outputs": {
            "comparison_csv": str(csv_path),
            "tng_manifest": str(output_dir / "tng_validation_manifest.json"),
            "bolshoi_manifest": str(output_dir / "halotools_bolshoi_ground_truth_manifest.json"),
        },
        "params": {
            "allow_network": bool(allow_network),
            "tng_base_path": str(tng_base_path),
            "k": int(k),
            "n_modes": int(n_modes),
            "bolshoi_max_n": int(bolshoi_max_n),
            "skip_tng": bool(skip_tng),
        },
    }

    (output_dir / "universality_ground_truth_suite_manifest.json").write_text(
        json.dumps(suite_manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return suite_manifest
