# Changelog

## 0.1.12 — 2026-01-02

This release adds a low-maintenance documentation site and strengthens “offline-first” onboarding and CI coverage.

- Add MkDocs documentation site (Material theme) and API reference pages.
- Add tiny committed example NPZ artifacts and generator script for offline demos.
- Expand CI to run tests on Ubuntu + Windows and add extras/docs smoke checks.

## 0.1.11 — 2026-01-01

- Tier-2 `fingerprint` now accepts cosmic NPZs without `quantum_mass` by computing it from `positions` (or `ra`+`dec`) when available.
- Fix Windows file-handle behavior by ensuring NPZ loads are closed promptly.

## 0.1.10 — 2026-01-01

- Add Tier-1 `add-quantum-mass` command to compute `quantum_mass` + eigenvalues from a local NPZ (positions or RA/Dec).
- Writes an output NPZ plus a JSON manifest (offline-first; no network).

## 0.1.9 — 2026-01-01

- Add Tier-2 `plot-16m` command (offline-first; writes manifest + results NPZ; PNG is optional if `matplotlib` is available).
- Fix Windows temp-file locking by closing input `.npz` handles promptly.
- Add unit test coverage and README usage example for the new command.
