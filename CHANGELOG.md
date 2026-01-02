# Changelog

## 0.1.9 — 2026-01-01

- Add Tier-2 `plot-16m` command (offline-first; writes manifest + results NPZ; PNG is optional if `matplotlib` is available).
- Fix Windows temp-file locking by closing input `.npz` handles promptly.
- Add unit test coverage and README usage example for the new command.
