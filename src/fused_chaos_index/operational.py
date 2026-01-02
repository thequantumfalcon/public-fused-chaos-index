from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import eigsh


FloatArray = NDArray[np.floating]


@dataclass(frozen=True)
class IPRResults:
    ipr_values: FloatArray
    mean_ipr: float
    localized_fraction: float


@dataclass(frozen=True)
class SFFResults:
    times: FloatArray
    sff_values: FloatArray
    ramp_slope: float


@dataclass(frozen=True)
class FCIResults:
    fci_normalized: float
    connectivity_measure: float
    chaos_measure: float
    physical_regime: str
    interpretation: str


class StreamlinedFCIPipeline:
    """Minimal operational pipeline: points → graph → spectrum → IPR/SFF → FCI.

    This is a cleaned, package-friendly extraction of the core logic from the
    research workspace.

    Dependencies are intentionally minimal: numpy + scipy.
    """

    def __init__(
        self,
        *,
        k_neighbors: int = 8,
        weight_scale: float = 10.0,
        disorder_strength: float = 0.1,
        seed: int = 42,
        H0_km_s_mpc: float = 70.0,
    ) -> None:
        self.k_neighbors = int(k_neighbors)
        self.weight_scale = float(weight_scale)
        self.disorder_strength = float(disorder_strength)
        self.seed = int(seed)

        self.H0 = float(H0_km_s_mpc)

    def simulate_euclid_like_catalog(self, *, n_galaxies: int = 5000) -> dict[str, FloatArray]:
        """Generate a small synthetic catalog in (ra, dec, z).

        Note: this is synthetic data for demonstration only.
        """

        rng = np.random.default_rng(self.seed)

        # Perseus-like anchor (just used as a plausible center)
        cluster_ra, cluster_dec, cluster_z = 49.95, 41.51, 0.017

        n_galaxies = int(n_galaxies)
        n_core = int(0.3 * n_galaxies)
        n_infall = int(0.4 * n_galaxies)
        n_field = n_galaxies - n_core - n_infall

        core_ra = rng.normal(cluster_ra, 0.05, n_core)
        core_dec = rng.normal(cluster_dec, 0.05, n_core)
        core_z = rng.normal(cluster_z, 0.001, n_core)

        infall_ra = rng.normal(cluster_ra, 0.2, n_infall)
        infall_dec = rng.normal(cluster_dec, 0.2, n_infall)
        infall_z = rng.normal(cluster_z + 0.003, 0.002, n_infall)

        field_ra = rng.uniform(cluster_ra - 0.4, cluster_ra + 0.4, n_field)
        field_dec = rng.uniform(cluster_dec - 0.4, cluster_dec + 0.4, n_field)
        field_z = rng.uniform(0.05, 0.3, n_field)

        ra = np.concatenate([core_ra, infall_ra, field_ra])
        dec = np.concatenate([core_dec, infall_dec, field_dec])
        z = np.concatenate([core_z, infall_z, field_z])

        return {"ra": ra.astype(np.float64), "dec": dec.astype(np.float64), "z": z.astype(np.float64)}

    def coordinates_to_comoving_cartesian(self, catalog: dict[str, FloatArray]) -> FloatArray:
        """Convert (ra, dec, z) to comoving Cartesian coordinates (Mpc).

        Uses the same simplified approximation from the research script:
        $D_c \approx (c z) / H_0$.
        """

        ra = np.asarray(catalog["ra"], dtype=np.float64)
        dec = np.asarray(catalog["dec"], dtype=np.float64)
        z = np.asarray(catalog["z"], dtype=np.float64)

        c_km_s = 299_792.458
        dc_mpc = (c_km_s * z) / self.H0

        ra_rad = np.radians(ra)
        dec_rad = np.radians(dec)

        x = dc_mpc * np.cos(dec_rad) * np.cos(ra_rad)
        y = dc_mpc * np.cos(dec_rad) * np.sin(ra_rad)
        z_cart = dc_mpc * np.sin(dec_rad)

        return np.column_stack([x, y, z_cart]).astype(np.float64)

    def build_knn_graph(self, positions: FloatArray) -> csr_matrix:
        """Construct a symmetric k-NN graph with exponential distance weights."""

        positions = np.asarray(positions, dtype=np.float64)
        n = int(positions.shape[0])

        tree = cKDTree(positions)
        distances, indices = tree.query(positions, k=self.k_neighbors + 1)

        rows: list[int] = []
        cols: list[int] = []
        weights: list[float] = []

        for i in range(n):
            for j in range(1, self.k_neighbors + 1):
                neighbor = int(indices[i, j])
                d = float(distances[i, j])
                w = float(np.exp(-d / self.weight_scale))
                rows.append(i)
                cols.append(neighbor)
                weights.append(w)

        adjacency = csr_matrix((weights, (rows, cols)), shape=(n, n), dtype=np.float64)
        adjacency = (adjacency + adjacency.T) / 2.0
        return adjacency

    def construct_anderson_hamiltonian(self, adjacency: csr_matrix) -> Any:
        """Anderson tight-binding Hamiltonian: H = diag(disorder) - A."""

        n = int(adjacency.shape[0])
        rng = np.random.default_rng(self.seed)
        disorder = rng.uniform(-self.disorder_strength, self.disorder_strength, n).astype(np.float64)
        return diags(disorder, 0, dtype=np.float64) - adjacency

    def compute_spectrum(self, hamiltonian: Any, *, n_eigenstates: int = 50) -> tuple[FloatArray, FloatArray]:
        """Compute the lowest-energy eigenpairs via sparse eigensolver."""

        n_eigenstates = int(n_eigenstates)
        evals, evecs = eigsh(hamiltonian, k=n_eigenstates, which="SA", tol=1e-6)
        return np.asarray(evals, dtype=np.float64), np.asarray(evecs, dtype=np.float64)

    def compute_ipr(self, eigenvectors: FloatArray) -> IPRResults:
        """Inverse Participation Ratio: IPR_α = Σ_i |ψ_α(i)|^4."""

        evecs = np.asarray(eigenvectors, dtype=np.float64)
        ipr_values = np.sum(np.abs(evecs) ** 4, axis=0).astype(np.float64)

        mean_ipr = float(np.mean(ipr_values))

        n_nodes = int(evecs.shape[0])
        delocalization_threshold = 5.0 / max(n_nodes, 1)
        localized_fraction = float(np.mean(ipr_values > delocalization_threshold))

        return IPRResults(ipr_values=ipr_values, mean_ipr=mean_ipr, localized_fraction=localized_fraction)

    def compute_sff(self, eigenvalues: FloatArray, *, max_time: float = 50.0, n_times: int = 30) -> SFFResults:
        """Spectral form factor (simple unfolding): SFF(t) = |Σ_j exp(-i E_j t)|^2."""

        evals = np.sort(np.asarray(eigenvalues, dtype=np.float64))
        n_levels = int(evals.shape[0])
        unfolded_levels = np.arange(n_levels, dtype=np.float64)

        times = np.linspace(0.5, float(max_time), int(n_times), dtype=np.float64)
        sff_values = np.zeros_like(times)

        for i, t in enumerate(times):
            phases = unfolded_levels * t
            z_t = np.sum(np.exp(-1j * phases))
            sff_values[i] = float(np.abs(z_t) ** 2)

        mid_start = len(times) // 3
        mid_end = 2 * len(times) // 3
        ramp_slope = float(np.polyfit(times[mid_start:mid_end], sff_values[mid_start:mid_end], 1)[0])

        return SFFResults(times=times, sff_values=sff_values, ramp_slope=ramp_slope)

    def compute_fused_chaos_index(self, ipr: IPRResults, sff: SFFResults) -> FCIResults:
        """FCI = (⟨IPR⟩^{-1}) / (ramp_slope), then normalized by sqrt(n_states)."""

        mean_ipr = float(ipr.mean_ipr)
        ramp_slope = float(max(sff.ramp_slope, 1e-6))

        connectivity = 1.0 / mean_ipr
        chaos = ramp_slope
        fci_raw = connectivity / chaos

        n_states = int(ipr.ipr_values.shape[0])
        fci_normalized = float(fci_raw / np.sqrt(max(n_states, 1)))

        lf = float(ipr.localized_fraction)
        if lf > 0.6:
            regime = "CLUSTER-DOMINATED (localized)"
            interpretation = "Strong cluster cores, weak filamentary connections"
        elif lf < 0.3:
            regime = "FILAMENT-DOMINATED (delocalized)"
            interpretation = "Strong cosmic web connectivity"
        else:
            regime = "INTERMEDIATE (mixed phase)"
            interpretation = "Balance between clusters and filaments"

        return FCIResults(
            fci_normalized=fci_normalized,
            connectivity_measure=float(connectivity),
            chaos_measure=float(chaos),
            physical_regime=regime,
            interpretation=interpretation,
        )

    def run(self, *, n_galaxies: int = 2000, n_eigenstates: int = 50) -> dict[str, Any]:
        """Run the full pipeline end-to-end and return structured results."""

        catalog = self.simulate_euclid_like_catalog(n_galaxies=int(n_galaxies))
        positions = self.coordinates_to_comoving_cartesian(catalog)
        adjacency = self.build_knn_graph(positions)
        hamiltonian = self.construct_anderson_hamiltonian(adjacency)

        n_eigs = min(int(n_eigenstates), int(positions.shape[0]) - 2)
        if n_eigs < 2:
            raise ValueError("n_galaxies too small to compute spectrum")

        eigenvalues, eigenvectors = self.compute_spectrum(hamiltonian, n_eigenstates=n_eigs)
        ipr = self.compute_ipr(eigenvectors)
        sff = self.compute_sff(eigenvalues)
        fci = self.compute_fused_chaos_index(ipr, sff)

        spectral_gap = float(np.sort(eigenvalues)[1] - np.sort(eigenvalues)[0])

        return {
            "n_galaxies": int(positions.shape[0]),
            "spectral_gap": spectral_gap,
            "ipr": ipr,
            "sff": sff,
            "fci": fci,
        }


def run_operational_demo(*, n_galaxies: int = 2000, k_neighbors: int = 10, seed: int = 42) -> FCIResults:
    pipeline = StreamlinedFCIPipeline(k_neighbors=k_neighbors, seed=seed)
    results = pipeline.run(n_galaxies=n_galaxies)
    return results["fci"]
