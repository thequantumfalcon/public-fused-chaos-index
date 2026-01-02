# Tier 2

Tier-2 commands are offline-first and artifact-driven.

## Universality sweep

```bash
fci tier2 universality-sweep --inputs run1.npz run2.npz --threshold 5e-7
```

## Fingerprint (Collatz vs cosmic)

```bash
fci tier2 fingerprint --collatz collatz_run.npz --smacs smacs_catalog.npz
```

If the SMACS NPZ does **not** contain `quantum_mass`, it will be computed automatically when `positions` or (`ra`+`dec`) are present:

```bash
fci tier2 fingerprint --collatz collatz_run.npz --smacs smacs_positions_kappa.npz --k 10 --n-modes 10
```

## Collatz summary

```bash
fci tier2 collatz-summary --collatz collatz_run.npz --baseline baseline_run.npz --runtime-seconds 136
```

## Plot 16M discovery

Writes a PNG only if `matplotlib` is available; always writes results NPZ + manifest.

```bash
fci tier2 plot-16m --collatz collatz_run.npz
```
