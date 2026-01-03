# Run Folders and Manifests

This document specifies the run folder structure and manifest schema for CLI commands.

## Overview

All CLI commands that write outputs follow a standardized contract for reproducibility and provenance tracking. This enables:
- Reproducible research workflows
- Artifact traceability
- Automated validation and testing
- Integration with external tools

## Run Folder Structure

### Creation

Every CLI command that writes outputs creates a timestamped run folder under the specified `--output-dir`:

```
<output-dir>/
  └── public_suite_20260103_143055/   # Timestamped run folder
      ├── manifest.json               # Run-level manifest (required)
      ├── tier2_path1_universality_manifest.json  # Per-command manifest
      └── tier2_path1_universality_results.npz    # Command outputs
```

The run folder name follows the format: `public_suite_YYYYMMDD_HHMMSS` (UTC timestamp).

### Run-Level Manifest

Every run folder MUST contain a `manifest.json` at its root. This provides execution context and metadata.

## Manifest Schema (Version 1)

### Required Fields

The run-level `manifest.json` must include the following top-level keys:

```json
{
  "manifest_schema_version": "1",
  "run_id": "public_suite_20260103_143055",
  "created_utc": "2026-01-03T14:30:55+00:00",
  "command": "fci tier2 universality-sweep --output-dir results --inputs data.npz",
  "package_version": "0.1.12",
  "environment": {
    "python_version": "3.12.0 (main, Oct 2 2023, 12:00:00)",
    "python_executable": "/usr/bin/python3",
    "platform": "Linux-6.5.0-1014-azure-x86_64-with-glibc2.35",
    "numpy_version": "1.26.2",
    "scipy_version": "1.11.4"
  },
  "status": "OK",
  "outputs": [
    {
      "name": "tier2_path1_universality_manifest",
      "path": "tier2_path1_universality_manifest.json",
      "type": "json"
    },
    {
      "name": "tier2_path1_universality_results",
      "path": "tier2_path1_universality_results.npz",
      "type": "npz"
    }
  ]
}
```

### Field Descriptions

#### `manifest_schema_version` (string, required)

The version of the manifest schema. Current version is `"1"`.

When the schema changes in a backwards-incompatible way, this version will be incremented.

#### `run_id` (string, required)

A unique identifier for this run. Typically matches the run folder name.

#### `created_utc` (string, required)

ISO 8601 timestamp of when the run was created, in UTC timezone.

Format: `YYYY-MM-DDTHH:MM:SS+00:00`

#### `command` (string, required)

The full CLI command that was executed. This can be used to reproduce the run.

Example: `"fci tier2 fingerprint --collatz data.npz --smacs catalog.npz"`

#### `package_version` (string, required)

The version of the `public-fused-chaos-index` package that was used.

Example: `"0.1.12"` or `"unknown"` if version cannot be determined.

#### `environment` (object, required)

Environment information for reproducibility. Required subfields:

- `python_version` (string): Full Python version string
- `python_executable` (string): Path to the Python interpreter
- `platform` (string): Platform identifier

Optional subfields (included if available):

- `numpy_version` (string): NumPy version
- `scipy_version` (string): SciPy version

#### `status` (string, required)

The outcome of the run. Must be one of:

- `"OK"`: Run completed successfully
- `"ERROR"`: Run encountered an error
- `"SKIP"`: Run was skipped (e.g., missing dependencies)

#### `outputs` (array, required)

List of output artifacts produced by this run. Each element is an object with:

- `name` (string): Human-readable artifact name (typically the filename without extension)
- `path` (string): Path to the artifact, relative to the run folder
- `type` (string): File type/extension (e.g., `"json"`, `"npz"`, `"png"`)

The `outputs` list may be empty for runs that produce no outputs.

### Optional Fields

Commands may include additional fields in the manifest. Consumers should gracefully ignore unknown fields.

## Per-Command Manifests

In addition to the run-level `manifest.json`, commands write per-command manifests that contain command-specific details:

- `tier2_path1_universality_manifest.json`
- `tier2_path2_fingerprint_manifest.json`
- `tier2_path3_stopping_time_vs_mass_manifest.json`
- etc.

These manifests are command-specific and are documented separately. They typically include:

- `experiment`: Human-readable experiment name
- `time_utc`: Timestamp
- `status`: Status code
- `params`: Parameters used for the run
- `inputs`: Input files with SHA256 hashes
- `results`: Summary statistics and findings
- `outputs`: Output file paths

## Compatibility Rules

### Additive Changes

The manifest schema allows **additive changes** without a version bump:

- New optional fields may be added
- New values in enumerated fields (e.g., new status codes) may be added
- Fields may become more detailed (e.g., adding more environment info)

Consumers must gracefully handle unknown fields and unknown enum values.

### Breaking Changes

Breaking changes require incrementing `manifest_schema_version`:

- Removing required fields
- Changing field types
- Changing field semantics
- Renaming fields

## Implementation Notes

### For CLI Developers

When adding new CLI commands that write outputs:

1. Use `default_run_dir(args.output_dir)` to create a timestamped run folder
2. Write command outputs to the run folder
3. Call `create_run_manifest()` after the command completes
4. Use `collect_outputs_from_dir()` to automatically discover outputs

Example:

```python
from .suite import default_run_dir
from .run_manifest import create_run_manifest, collect_outputs_from_dir

run_dir = default_run_dir(args.output_dir)

# Run the command, writing outputs to run_dir
result = run_my_command(output_dir=run_dir, ...)

# Create the run-level manifest
outputs = collect_outputs_from_dir(run_dir)
create_run_manifest(
    run_dir=run_dir,
    command=cli_command,
    status=result.get("status", "OK"),
    outputs=outputs,
)
```

### For Consumers

When consuming manifests:

1. Check `manifest_schema_version` - only parse versions you understand
2. Validate required fields are present
3. Gracefully handle unknown fields
4. Use `outputs` list to discover available artifacts
5. Read per-command manifests for detailed results

## Testing

Tests validate the manifest contract:

- `tests/test_cli_manifest.py`: Tests that CLI commands create valid manifests
- Each test runs a CLI command and validates:
  - Run folder creation
  - `manifest.json` presence and structure
  - Required fields and types
  - Output files referenced in the manifest exist

Run the tests:

```bash
python -m unittest tests.test_cli_manifest -v
```
