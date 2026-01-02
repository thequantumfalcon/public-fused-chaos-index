# NPZ Contracts

This project uses `.npz` artifacts as the primary offline-first exchange format.

## Common keys

### Quantum mass

Many Tier-2 tools accept one of:
- `quantum_mass` (preferred)
- `mass`
- `M`

### Positions

Some tools can compute `quantum_mass` if one of the following exists:
- `positions` (N×D)
- `ra` + `dec` (degrees)

### Lensing signal

Cosmic/candidate catalogs commonly use:
- `kappa` (N,)

## Notes
- Tools should not download data implicitly.
- Tools should close `.npz` handles promptly (important on Windows).
