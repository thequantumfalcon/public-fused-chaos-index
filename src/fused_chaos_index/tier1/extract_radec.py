from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]

_RA_CANDIDATES: tuple[str, ...] = (
    "ra",
    "raj2000",
    "alpha_j2000",
    "alpha",
    "coord_ra",
    "ra_deg",
    "ra2000",
    "ra_icrs",
    "ra_j2000",
)

_DEC_CANDIDATES: tuple[str, ...] = (
    "dec",
    "dej2000",
    "delta_j2000",
    "delta",
    "coord_dec",
    "dec_deg",
    "dec2000",
    "dec_icrs",
    "dec_j2000",
)


def _norm(name: str) -> str:
    return name.strip().lower().replace(" ", "").replace("-", "_")


def _pick_col(all_cols: Iterable[str], preferred: Optional[str], candidates: tuple[str, ...]) -> str:
    cols = list(all_cols)
    if preferred:
        preferred_n = _norm(preferred)
        for c in cols:
            if _norm(c) == preferred_n:
                return c
        raise ValueError(f"Requested column not found: {preferred}. Available: {cols}")

    norm_map = {_norm(c): c for c in cols}
    for cand in candidates:
        if cand in norm_map:
            return norm_map[cand]

    raise ValueError(
        "Could not auto-detect RA/Dec columns. "
        f"Available columns: {cols}. "
        "Pass ra_col/dec_col explicitly."
    )


def _maybe_hours_to_deg(ra: FloatArray, dec: FloatArray) -> tuple[FloatArray, str]:
    ra_min = float(np.nanmin(ra))
    ra_max = float(np.nanmax(ra))
    dec_min = float(np.nanmin(dec))
    dec_max = float(np.nanmax(dec))

    if 0.0 <= ra_min and ra_max <= 24.5 and -90.0 <= dec_min <= 90.0 and -90.0 <= dec_max <= 90.0:
        return (ra * 15.0).astype(np.float64), "hours->deg"
    return ra.astype(np.float64), "deg"


def extract_radec_to_npz(
    *,
    catalog_path: Path,
    out_npz: Path,
    ra_col: str | None = None,
    dec_col: str | None = None,
    max_rows: int = 0,
) -> dict[str, object]:
    """Extract RA/Dec columns from an Astropy-readable catalog into a compressed NPZ.

    Requires optional dependency: `astropy`.
    """

    if not catalog_path.exists():
        raise FileNotFoundError(str(catalog_path))

    try:
        from astropy.table import Table
        from astropy.io.registry import IORegistryError
    except Exception as e:  # pragma: no cover
        raise RuntimeError("This command requires the 'astro' extra: pip install -e .[astro]") from e

    try:
        tab = Table.read(str(catalog_path))
    except IORegistryError:
        raw = np.genfromtxt(str(catalog_path), comments="#", dtype=np.float64)
        if raw.ndim != 2 or raw.shape[1] < 3:
            raise ValueError(f"Catalog format not recognized and numeric fallback failed: shape={getattr(raw, 'shape', None)}")

        ra = raw[:, 1].astype(np.float64, copy=False)
        dec = raw[:, 2].astype(np.float64, copy=False)
    else:
        if len(tab) == 0:
            raise ValueError("Catalog table is empty")

        ra_name = _pick_col(tab.colnames, ra_col, _RA_CANDIDATES)
        dec_name = _pick_col(tab.colnames, dec_col, _DEC_CANDIDATES)
        ra = np.asarray(tab[ra_name], dtype=np.float64)
        dec = np.asarray(tab[dec_name], dtype=np.float64)

    if max_rows and int(max_rows) > 0:
        ra = ra[: int(max_rows)]
        dec = dec[: int(max_rows)]

    finite = np.isfinite(ra) & np.isfinite(dec)
    ra = ra[finite]
    dec = dec[finite]

    if ra.size == 0:
        raise ValueError("No finite RA/Dec rows found")

    ra, unit_note = _maybe_hours_to_deg(ra, dec)

    if not (np.nanmin(dec) >= -90.0 and np.nanmax(dec) <= 90.0):
        raise ValueError("Dec values out of [-90, 90] after parsing; check catalog")

    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_npz, ra=ra.astype(np.float64), dec=dec.astype(np.float64))

    return {
        "out_npz": str(out_npz),
        "n": int(ra.size),
        "ra_units": unit_note,
        "ra_range": (float(np.nanmin(ra)), float(np.nanmax(ra))),
        "dec_range": (float(np.nanmin(dec)), float(np.nanmax(dec))),
    }
