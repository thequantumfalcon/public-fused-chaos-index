from __future__ import annotations

from pathlib import Path

import numpy as np


def main() -> None:
    base = Path(__file__).resolve().parent / "data"
    base.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(0)

    # Collatz-like artifact: quantum_mass + eigenvalues
    mass = np.abs(rng.normal(size=(500,))).astype(np.float64) + 1e-6
    mass[:10] = rng.uniform(1e-9, 1e-7, size=(10,)).astype(np.float64)
    eigs = np.sort(np.abs(rng.normal(size=(10,))).astype(np.float64) + 1e-3)
    np.savez_compressed(base / "collatz_example.npz", quantum_mass=mass, eigenvalues=eigs)

    # Cosmic artifact: positions + kappa (no quantum_mass; Tier-2 fingerprint will compute it)
    positions = rng.normal(size=(400, 3)).astype(np.float64)
    kappa = (0.5 * rng.normal(size=(400,)) + 0.1).astype(np.float64)
    np.savez_compressed(base / "smacs_positions_kappa_example.npz", positions=positions, kappa=kappa)

    # Baseline artifact for collatz-summary
    np.savez_compressed(base / "baseline_example.npz", quantum_mass=mass * 1.05)

    print("Wrote:")
    for p in sorted(base.glob("*.npz")):
        print(" -", p.name, f"({p.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
