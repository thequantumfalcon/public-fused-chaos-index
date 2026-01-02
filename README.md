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

## Install

```bash
pip install -e .
```

Optional extras (only needed for certain commands):

```bash
pip install -e ".[astro,viz,ml]"
```

## Quick start

Discover commands:

```bash
fci --help
```

Offline-first suite (recommended first run):

```bash
fci suite run
```

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
fci tier1 score-single --prediction-card-json card.json --artifact-json artifact.json
fci tier1 score-frontier --prediction-card card.json --frontier-manifest frontier_manifest.json
fci gate
```

## Run tests

```bash
python -m unittest discover -s tests -p "test*.py"
```

## Scientific note

This code computes spectral / graph-derived quantities. Any downstream scientific interpretation (e.g., cosmological parameter constraints) requires careful validation and should not be treated as established causal proof.
