# Manifests

Most commands write a JSON manifest alongside results.

Typical fields:
- `experiment`: human-readable name
- `time_utc`: timestamp
- `status`: `OK` | `SKIP` | `ERROR`
- `inputs`: input paths + SHA256 hashes
- `params`: parameters used
- `results`: summary statistics
- `outputs`: output file paths

Manifests are intended to be stable enough for automation and review.
