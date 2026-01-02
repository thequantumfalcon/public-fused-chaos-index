# Examples

This repo includes tiny offline example artifacts under `examples/data/`.

## Tier-2 fingerprint (uses SMACS positions+kappa)

```bash
fci tier2 fingerprint \
  --collatz examples/data/collatz_example.npz \
  --smacs examples/data/smacs_positions_kappa_example.npz \
  --k 10 --n-modes 10
```

## Tier-2 collatz summary

```bash
fci tier2 collatz-summary \
  --collatz examples/data/collatz_example.npz \
  --baseline examples/data/baseline_example.npz \
  --runtime-seconds 136
```
