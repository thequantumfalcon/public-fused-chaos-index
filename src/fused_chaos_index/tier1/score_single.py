from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def score_single_artifact(*, prediction_card_json: Path, artifact_json: Path) -> dict[str, Any]:
    card = _read_json(prediction_card_json)
    art = _read_json(artifact_json)

    thresholds = card.get("primary_test", {})
    p_thr = float(thresholds.get("p_threshold", 1e-6))
    abs_rho_thr = float(thresholds.get("abs_rho_threshold", 0.15))

    status = art.get("status") if isinstance(art.get("status"), dict) else {"ok": True}
    status_ok = bool(status.get("ok", True))

    corr = art.get("correlation") if isinstance(art.get("correlation"), dict) else {}
    rho_raw = corr.get("spearman_rho")
    p_raw = corr.get("spearman_p")

    rho: float | None
    p: float | None

    if (not status_ok) or (rho_raw is None) or (p_raw is None):
        rho = None
        p = None
        passed = False
    else:
        rho = float(rho_raw)
        p = float(p_raw)
        passed = (p < p_thr) and (abs(rho) >= abs_rho_thr)

    return {
        "dataset": card.get("dataset"),
        "decision": {
            "passed": bool(passed),
            "status": {
                "ok": bool(status_ok),
                "reason": status.get("reason"),
                "message": status.get("message"),
            },
            "rule": {
                "p_lt": p_thr,
                "abs_rho_ge": abs_rho_thr,
            },
            "observed": {
                "spearman_rho": rho,
                "spearman_p": p,
            },
        },
        "notes": (
            "PASS/FAIL computed from frozen Tier-1 thresholds. "
            "Interpretation depends on whether positions came from a real catalog/arcs vs pixel sampling."
        ),
        "inputs": {
            "prediction_card_json": str(prediction_card_json),
            "artifact_json": str(artifact_json),
        },
    }
