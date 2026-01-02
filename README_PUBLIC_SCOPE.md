# Public scope

This folder is a **public-clean** release: it aims to be broadly usable while staying **low-friction**.

Included:
- Core FCI computation (streamlined operational pipeline)
- SYK–Collatz constant approximation function
- Meta-tools (evidence aggregation + robustness diagnostics)
- Tier-1 helpers (catalog RA/Dec extraction, scoring utilities)
- Offline-first suite runner (`fci suite run`) that writes a single manifest
- Falsification/null-test “gate” for suite manifests

Excluded (by design):
- Large data artifacts (`.npz`, `.fits`, `.csv` bundles, etc.)
- Default-on web scrapers / downloaders (anything networked should be explicit opt-in)
- Internal research notes, one-off experiments, GPU scaling campaigns, archives

Principle:
- If it requires heavy datasets or network access, it should be an optional add-on step, not a default public install/run.
