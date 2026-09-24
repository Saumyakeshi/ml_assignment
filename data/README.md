# Data Directory

| Directory | Purpose | Version control |
|---|---|---|
| `raw/` | Original, immutable datasets | Generated contents are ignored |
| `processed/` | Final model-ready features and splits | Generated contents are ignored |

For each dataset, record its source, license, download date, checksum if available, target variable, and known limitations in `docs/research/`.

Do not overwrite raw data. Keep every transformation reproducible in the relevant self-contained notebook.
