from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def check_frontier_manifest(path: Path) -> list[str]:
    issues: list[str] = []
    data = _read_json(path)

    clusters = data.get("clusters")
    if not isinstance(clusters, list) or not clusters:
        return ["frontier: missing or empty 'clusters' list"]

    for c in clusters:
        cid = str(c.get("cluster_id", "?"))
        null = c.get("null_test")
        if not isinstance(null, dict):
            issues.append(f"frontier[{cid}]: missing null_test")
            continue

        if "p_permutation" not in null:
            issues.append(f"frontier[{cid}]: null_test missing p_permutation")
            continue

        try:
            p = float(null["p_permutation"])
        except Exception:
            issues.append(f"frontier[{cid}]: p_permutation not a number")
            continue

        if not (0.0 <= p <= 1.0):
            issues.append(f"frontier[{cid}]: p_permutation out of [0,1]")

        corr = c.get("correlation")
        if not isinstance(corr, dict) or "spearman_rho" not in corr or "spearman_p" not in corr:
            issues.append(f"frontier[{cid}]: correlation missing spearman_rho/spearman_p")

        prov = c.get("provenance")
        if not isinstance(prov, dict) or "kappa_sha256" not in prov:
            issues.append(f"frontier[{cid}]: provenance missing kappa_sha256")

    return issues


def check_universality_manifest(path: Path) -> list[str]:
    issues: list[str] = []
    data = _read_json(path)

    results = data.get("results")
    if not isinstance(results, dict):
        return ["universality: missing results dict"]

    if "bolshoi" not in results:
        issues.append("universality: missing bolshoi result")
    else:
        bol = results["bolshoi"]
        if not isinstance(bol, dict) or bol.get("status") not in {"OK", "SKIP", "ERROR"}:
            issues.append("universality: bolshoi status missing/invalid")

    return issues
