# Data Directory

| Directory | Purpose | Version control |
|---|---|---|
| `raw/` | Original, immutable datasets | Ignored except `.gitkeep` |
| `external/` | Third-party supporting data | Ignored except `.gitkeep` |
| `interim/` | Cleaned or partially transformed data | Ignored except `.gitkeep` |
| `processed/` | Final model-ready features and splits | Ignored except `.gitkeep` |
| `samples/` | Small anonymized fixtures for tests and examples | Tracked |

For each dataset, record its source, license, download date, checksum if available, target variable, and known limitations in `docs/research/`.

Do not overwrite raw data. Make transformations reproducible through code in `src/comp70049/` or `scripts/`.

