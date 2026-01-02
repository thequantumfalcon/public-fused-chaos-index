# Release process

This repo uses simple git tags.

## Local steps

1. Ensure tests are green:

```bash
python -m unittest discover -s tests -p "test*.py"
```

2. Bump the version in `pyproject.toml` (and `CITATION.cff` if you keep it in sync).
3. Update `CHANGELOG.md`.
4. Commit and tag:

```bash
git commit -am "Release X.Y.Z: ..."
git tag -a vX.Y.Z -m "vX.Y.Z"
```

5. Push:

```bash
git push origin main --tags
```

## GitHub release

Create a GitHub Release from the new tag and paste the changelog entry.
