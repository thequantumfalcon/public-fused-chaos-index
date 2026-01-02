from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Thresholds:
    p_threshold: float
    abs_rho_threshold: float


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_thresholds(card_path: Path) -> Thresholds:
    card = _read_json(card_path)
    primary = card.get("primary_test", {})

    if primary.get("statistic") not in (None, "spearman"):
        raise ValueError("Prediction card primary_test.statistic is not 'spearman'")

    return Thresholds(
        p_threshold=float(primary.get("p_threshold")),
        abs_rho_threshold=float(primary.get("abs_rho_threshold")),
    )


def score_frontier_manifest(*, manifest_path: Path, thresholds: Thresholds) -> dict[str, Any]:
    m = _read_json(manifest_path)
    clusters = m.get("clusters")
    if not isinstance(clusters, list) or not clusters:
        raise ValueError("Frontier manifest missing or empty 'clusters' list")

    rows: list[dict[str, Any]] = []
    n_pass = 0

    for c in clusters:
        cid = str(c.get("cluster_id", "?"))
        name = str(c.get("cluster_name", "?"))
        corr = c.get("correlation", {})

        rho = float(corr.get("spearman_rho"))
        p = float(corr.get("spearman_p"))

        passed = (abs(rho) >= thresholds.abs_rho_threshold) and (p < thresholds.p_threshold)
        n_pass += int(passed)

        rows.append(
            {
                "cluster_id": cid,
                "cluster_name": name,
                "spearman_rho": rho,
                "spearman_p": p,
                "pass": bool(passed),
            }
        )

    return {
        "thresholds": {
            "p_threshold": thresholds.p_threshold,
            "abs_rho_threshold": thresholds.abs_rho_threshold,
        },
        "n_total": int(len(rows)),
        "n_pass": int(n_pass),
        "pass_rate": float(n_pass) / float(len(rows)),
        "per_cluster": rows,
        "notes": (
            "Scored against Tier-1 frozen thresholds from a prereg prediction card. "
            "PASS requires both p < p_threshold and |rho| >= abs_rho_threshold. "
            "Empirical/statistical association only; not causality."
        ),
    }
