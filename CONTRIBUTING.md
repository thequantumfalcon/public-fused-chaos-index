# Contributing

Thanks for your interest in contributing.

## Setup

```bash
pip install -e .
```

Optional extras (only if you use those commands):

```bash
pip install -e ".[astro,viz,ml]"
```

## Run tests

```bash
python -m unittest discover -s tests -p "test*.py"
```

## Style

Keep changes small and focused. Prefer:
- Pure functions and deterministic outputs where possible
- Lazy imports for optional dependencies
- No large binary/data files in the repo
