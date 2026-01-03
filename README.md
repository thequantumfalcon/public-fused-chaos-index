# Public Fused Chaos Index

This repository is a public-clean release of the **Fused Chaos Index (FCI)** codebase.

Goal: keep it **pristine** (package + CLI + docs), while excluding heavyweight artifacts (large data files, caches, internal notes, and one-off experiments).

Included:
- A streamlined operational pipeline: galaxy positions → k-NN graph → sparse Hamiltonian spectrum → IPR + SFF → FCI.
- The SYK–Collatz “constant approximation” computation as a small reusable function.
- Meta-tools: evidence aggregation + robustness diagnostics.
- Tier-1 helpers: RA/Dec extraction utility, scoring utilities, and a falsification/null-test “gate”.

Not included:
- Large datasets (`.npz`, `.fits`, etc.), scraping/downloader scripts, internal research notes, or one-off experiments.
	- Exception: a few tiny example artifacts are versioned under `examples/data/` for offline demos.

Docs site: https://thequantumfalcon.github.io/public-fused-chaos-index/

## Install

```bash
pip install -e .
```

Optional extras (only needed for certain commands):

```bash
pip install -e ".[astro,viz,ml,sim]"
```

Or install everything optional:

```bash
pip install -e ".[all]"
```

Docs site (MkDocs):

```bash
pip install -e ".[docs]"
python -m mkdocs serve
```

Docs maintenance (minimal):
- Edit Markdown in `docs/`
- Update navigation in `mkdocs.yml`
- Preview locally with `python -m mkdocs serve`
- CI (and Pages deploy, if enabled) runs `python -m mkdocs build --strict`

## Quick start

Discover commands:

```bash
fci --help
```

Offline-first suite (recommended first run):

```bash
fci suite run --profile offline
```

Suite profiles:

```bash
fci suite run --profile smoke
fci suite run --profile offline
fci suite run --profile full
```

Pass-through options for the universality step (used in `offline`/`full` profiles):

```bash
fci suite run --profile full --tng-base-path path/to/TNG300-1/output
fci suite run --profile full --tng-base-path path/to/illustris_validation_results.npz
fci suite run --profile full --skip-tng
```

Universality ground-truth suite (safe: SKIP if deps/data missing):

```bash
fci suite universality
```

Frontier evidence suite (requires local artifacts):

```bash
fci suite frontier --clusters-json path/to/clusters.json
```

`clusters.json` format:

```json
{
	"clusters": [
		{
			"cluster_id": "a2744",
			"cluster_name": "Abell 2744",
			"npz_path": "a2744_artifact.npz"
		}
	]
}
```

Each `.npz` must contain `positions` (N×D) and `kappa` (N).

Operational (synthetic Euclid-like catalog):

```bash
fci operational --n-galaxies 2000 --k 10
```

SYK–Collatz constant approximation:

```bash
fci syk-collatz
```

Meta tools:

```bash
fci meta audit "some text"
fci meta combine 0.7 0.8 0.9 --prior 0.5
fci meta ablation 1 2 3 4 5 --y 1 2 3 4 5
```

Tier-1 helpers:

```bash
fci tier1 extract-radec --catalog path/to/catalog.fits --out-npz catalog_radec.npz
fci tier1 add-quantum-mass --input-npz catalog.npz --out-npz catalog_with_qm.npz --k 10 --n-modes 10
fci tier1 score-single --prediction-card-json card.json --artifact-json artifact.json
fci tier1 score-frontier --prediction-card card.json --frontier-manifest frontier_manifest.json
fci gate
```

Tier-2 analyses (artifact-driven, offline-first):

```bash
# Tier-2 Path 1: compare multiple quantum-mass artifacts under one threshold
fci tier2 universality-sweep --inputs run1.npz run2.npz --threshold 5e-7

# Tier-2 Path 2: cosmic vs Collatz fingerprint (distribution fingerprints + correlations)
fci tier2 fingerprint --collatz collatz_run.npz --smacs smacs_catalog.npz

# If your SMACS NPZ lacks `quantum_mass`, the command will compute it if `positions`
# (or `ra`+`dec`) are present; tune the solver with:
fci tier2 fingerprint --collatz collatz_run.npz --smacs smacs_positions_kappa.npz --k 10 --n-modes 10

# Tier-2: Collatz NPZ summary (manuscript-ready stats + spectral gap ratio if eigenvalues are present)
fci tier2 collatz-summary --collatz collatz_run.npz --baseline baseline_run.npz --runtime-seconds 136

# Tier-2 Path 3: stopping time vs quantum-mass association test
fci tier2 stopping-time --input collatz_run.npz --sample-size 200000 --max-steps 5000

# Tier-2: optional 16M discovery plot (writes manifest/results; PNG only if matplotlib is available)
fci tier2 plot-16m --collatz collatz_run.npz
```

Notes:
- Inputs must contain one of `quantum_mass` (preferred), `mass`, or `M`.
- Outputs are written into a timestamped run folder under `--output-dir` as NPZ + JSON manifest.

## Reproducible outputs contract

All CLI commands that write outputs follow a standardized contract for reproducibility and provenance tracking:

**Run folder structure:**
- Each CLI command that writes outputs creates a timestamped run folder under `--output-dir` (e.g., `public_suite_20260103_143055`)
- The run folder contains all artifacts produced by that command invocation

**Run-level manifest (`manifest.json`):**
- Every run folder contains a `manifest.json` at its root
- This manifest provides execution context and metadata:
  - `manifest_schema_version`: Schema version (currently "1")
  - `run_id`: Unique identifier (typically the run folder name)
  - `created_utc`: Timestamp of execution
  - `command`: Full CLI command that was executed
  - `package_version`: Version of public-fused-chaos-index
  - `environment`: Python, numpy, scipy versions, and platform info
  - `status`: Run status (`OK`, `ERROR`, or `SKIP`)
  - `outputs`: List of output artifacts with name, relative path, and type

**Per-command manifests:**
- Commands also write per-command manifests (e.g., `tier2_path1_universality_manifest.json`)
- These contain command-specific details, results, and statistics
- Per-command manifests complement the run-level manifest

**Compatibility:**
- The contract allows additive changes (new fields)
- Breaking changes require a schema version bump
- Tools consuming manifests should gracefully handle unknown fields

See [docs/artifacts/run_folders.md](docs/artifacts/run_folders.md) for the detailed manifest schema specification.

Validators (safe: SKIP if deps/data missing):

```bash
fci validate bolshoi
fci validate tng --base-path path/to/TNG300-1/output
fci validate tng --base-path path/to/derived_tng_like_artifact.npz
```

Notes:
- `bolshoi` requires `pip install -e ".[sim]"` and will not download unless you pass `--allow-network`.
- `tng` supports two input modes:
	- Raw mode: a local TNG group catalog and the `illustris_python` package (not bundled as a dependency).
	- Derived NPZ mode: a `.npz` containing `positions` (N×D) and `mass` (N), for offline-safe validation without bundling large raw catalogs.

## Run tests

```bash
python -m unittest discover -s tests -p "test*.py"
```

## Scientific note

This code computes spectral / graph-derived quantities. Any downstream scientific interpretation (e.g., cosmological parameter constraints) requires careful validation and should not be treated as established causal proof.
