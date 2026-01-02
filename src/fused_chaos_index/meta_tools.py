from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


def keyword_risk_audit(text: str, hazard_keywords: Sequence[str] | None = None) -> tuple[str, list[str]]:
    """Flag simple dual-use keywords in free-form text.

    Conservative preflight filter; not a substitute for policy/expert review.
    """

    if hazard_keywords is None:
        hazard_keywords = ("cryptography", "weapon", "exploitation", "nuclear", "biological")

    normalized = "".join(ch.lower() if ch.isalnum() or ch.isspace() else " " for ch in text)
    normalized = " ".join(normalized.split())

    matched = [kw for kw in hazard_keywords if kw.lower() in normalized]
    return ("Flagged" if matched else "Cleared"), matched


def combine_evidence_heuristic(evidences: Iterable[float], prior: float = 0.5) -> float:
    """Heuristically combine multiple evidence strengths into a single score.

    Implements a sequential Bayes-style update on odds, treating each `ev` as an
    evidence strength in [0,1]. Without calibration, this is not a probability.
    """

    posterior = float(np.clip(prior, 0.0, 1.0))
    for ev in evidences:
        e = float(np.clip(ev, 0.0, 1.0))
        denom = (e * posterior) + ((1.0 - e) * (1.0 - posterior))
        if denom <= 0.0:
            return float(posterior)
        posterior = (e * posterior) / denom
    return float(posterior)


@dataclass(frozen=True)
class AblationSensitivityResult:
    baseline: float
    mean_ablated: float
    mean_drop: float
    frac_drop_gt_10pct: float


def _corr(x: FloatArray, y: FloatArray, method: str) -> float:
    if method == "pearson":
        x0 = x - float(np.mean(x))
        y0 = y - float(np.mean(y))
        denom = float(np.sqrt(np.sum(x0 * x0) * np.sum(y0 * y0)))
        if denom == 0.0:
            return float("nan")
        return float(np.sum(x0 * y0) / denom)

    if method == "spearman":
        xr = np.argsort(np.argsort(x)).astype(np.float64)
        yr = np.argsort(np.argsort(y)).astype(np.float64)
        return _corr(xr, yr, method="pearson")

    raise ValueError("method must be 'pearson' or 'spearman'")


def ablation_sensitivity(
    x: FloatArray,
    y: FloatArray,
    *,
    n_trials: int = 200,
    method: str = "spearman",
    seed: int = 42,
) -> AblationSensitivityResult:
    """Measure robustness of a correlation to single-point ablations."""

    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape")

    n = int(x.shape[0])
    if n < 10:
        raise ValueError("need at least 10 samples for a meaningful diagnostic")

    baseline = _corr(x, y, method)
    if not np.isfinite(baseline):
        raise ValueError("baseline correlation is not finite")

    rng = np.random.default_rng(seed)
    x_mean = float(np.mean(x))

    ablated_corrs = np.empty(int(n_trials), dtype=np.float64)
    for t in range(int(n_trials)):
        idx = int(rng.integers(0, n))
        x_abl = x.copy()
        x_abl[idx] = x_mean
        ablated_corrs[t] = _corr(x_abl, y, method)

    mean_abl = float(np.mean(ablated_corrs))
    mean_drop = float(baseline - mean_abl)

    drop = baseline - ablated_corrs
    thresh = 0.1 * abs(baseline)
    frac = float(np.mean(np.abs(drop) > thresh))

    return AblationSensitivityResult(
        baseline=float(baseline),
        mean_ablated=mean_abl,
        mean_drop=mean_drop,
        frac_drop_gt_10pct=frac,
    )
