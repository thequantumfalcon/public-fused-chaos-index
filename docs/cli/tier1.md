# Tier 1

## Extract RA/Dec

```bash
fci tier1 extract-radec --catalog path/to/catalog.fits --out-npz catalog_radec.npz
```

## Add quantum mass to a catalog

Computes `quantum_mass` + `eigenvalues` and writes an output NPZ plus a JSON manifest.

```bash
fci tier1 add-quantum-mass --input-npz catalog.npz --out-npz catalog_with_qm.npz --k 10 --n-modes 10
```

Input NPZ must contain either:
- `positions` (N×D), or
- `ra` + `dec` (degrees)
