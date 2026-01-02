# Maintaining the Docs (MkDocs)

This repo’s docs are meant to be low-effort:
- Content lives in `docs/` as Markdown.
- Navigation lives in `mkdocs.yml`.
- API pages are generated automatically via `mkdocstrings` (they render Python docstrings).

## The only commands you need

Preview locally (live reload):

```bash
pip install -e ".[docs]"
python -m mkdocs serve
```

Build once (same as CI):

```bash
python -m mkdocs build --strict
```

## One-time GitHub setup (for auto-publishing)

If you want the docs to publish automatically, enable GitHub Pages to use Actions:
Settings → Pages → Source = **GitHub Actions**.

## Common tasks

### Add a new page

1. Create a new Markdown file under `docs/`, e.g. `docs/new_page.md`.
2. Add it to the `nav:` section of `mkdocs.yml`.
3. Run `python -m mkdocs serve` and click around.

### Update CLI/examples text

- Update the relevant Markdown page under `docs/cli/` or `docs/examples.md`.
- Keep examples offline-first (local artifacts only).

### Update API reference

- Update docstrings in the Python code under `src/fused_chaos_index/`.
- The API pages in `docs/api/` use `mkdocstrings` directives like:

```markdown
::: fused_chaos_index.tier2
```

No manual syncing is required beyond keeping imports working.

## If the docs build fails

- “Page not found” / broken nav: check `mkdocs.yml` paths.
- “mkdocstrings” import error: ensure the module name in `docs/api/*.md` matches real Python modules.
- CI uses `python -m mkdocs ...` to avoid PATH issues, especially on Windows.
