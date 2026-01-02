# Suites & Validators

## Suites

```bash
fci suite run --profile smoke
fci suite run --profile offline
fci suite run --profile full
```

## Validators

Validators should be SKIP-safe when optional deps/data are missing.

```bash
fci validate bolshoi
fci validate tng --base-path path/to/TNG300-1/output
```
