# Section 3 Demonstration - Anomaly Detection for Cyber Threats

## Assignment alignment

This experiment implements the Section 3 requirements with:

- **Dataset:** unclean CSE-CIC-IDS2018 flow data from Thursday 1 March 2018;
- **Classical model:** Isolation Forest;
- **Deep-learning model:** dense Autoencoder;
- **Training:** benign traffic only;
- **Threshold selection:** maximum F1 on labelled validation scores;
- **Required evaluation:** true-positive rate, false-positive rate, and precision-recall curves.

The implementation also reports precision, F1, average precision, ROC AUC, confusion matrices, score distributions, and Autoencoder training loss.

## Data download

From the repository root:

```powershell
.\.venv\Scripts\python.exe .\scripts\download_cic_ids2018.py
```

The script downloads one official 103 MiB CSV from the public CSE-CIC-IDS2018 AWS bucket, verifies its byte size and SHA-256 checksum, and saves it under the Git-ignored `data/raw/cic-ids2018/` directory.

## Cleaning performed

The initial audit proves that the selected file is unclean. The reproducible pipeline:

1. strips leading/trailing whitespace from column names and labels;
2. removes 25 embedded repeated CSV headers;
3. removes the 73 remaining exact duplicate records (the raw audit reports 97 duplicates because 24 repeated headers duplicate one another);
4. replaces 4,000 positive/negative infinite values with missing values;
5. coerces any malformed numeric values to missing values;
6. derives `hour` and `day_of_week` while reporting invalid timestamps;
7. drops features only when their missing fraction exceeds the configured limit;
8. median-imputes the remaining missing values;
9. removes zero-variance transformed features; and
10. standardizes numerical features using statistics learned from benign training data only.

Statistical outliers are retained because deleting them could remove the attacks that anomaly detection is intended to find.

## Leakage-safe anomaly experiment

The split differs from ordinary supervised classification:

- **Training:** benign flows only; labels are not used for model fitting.
- **Validation:** benign plus infiltration flows; labels select each score threshold.
- **Test:** held-out benign plus infiltration flows; used once for final evaluation.

The default sample produces 72,000 training, 33,000 validation, and 33,000 test records. Configuration is stored in [`configs/section_03.json`](../../configs/section_03.json).

## Models

### Isolation Forest

Isolation Forest uses random partitioning trees. Sparse observations generally require fewer splits and receive more anomalous scores. The implementation uses 200 trees with up to 10,000 samples per tree and does not use the model's built-in contamination threshold; it selects a threshold on validation data instead.

### Autoencoder

The Autoencoder learns to reconstruct standardized benign traffic through a compressed latent layer. Mean squared reconstruction error is the anomaly score. It uses a `64 -> 32 -> 12 -> 32 -> 64` hidden/latent structure around the input/output layers, dropout, Adam optimization, and early stopping based only on benign validation reconstruction loss.

## Run locally

Install dependencies and execute:

```powershell
python -m pip install -e ".[deep,dev]"
.\.venv\Scripts\python.exe .\scripts\run_section_03.py
```

For a one-epoch pipeline check:

```powershell
.\.venv\Scripts\python.exe .\scripts\run_section_03.py --epochs 1
```

The executable Colab workflow is in [`notebooks/03_anomaly_detection/section-03-anomaly-detection.ipynb`](../../notebooks/03_anomaly_detection/section-03-anomaly-detection.ipynb). It clones the repository automatically in a fresh hosted runtime.

## Full configured result

The checked-in artifacts come from the configured 12-epoch Autoencoder run:

| Model | TPR | FPR | Precision | F1 | Average precision | ROC AUC |
|---|---:|---:|---:|---:|---:|---:|
| Isolation Forest | 0.956 | 0.916 | 0.281 | 0.435 | 0.247 | 0.461 |
| Autoencoder (12 epochs) | 0.997 | 0.888 | 0.296 | 0.457 | 0.277 | 0.517 |

These numbers are deliberately not presented as a successful detector. Maximizing validation F1 selected very permissive thresholds because the scores poorly separate infiltration from benign traffic. The models catch most attacks only by creating an operationally unacceptable number of false alerts. The Autoencoder reduced FPR relative to its one-epoch smoke run, but the completed training still does not establish that this representation reliably separates stealthy infiltration flows.

## Interpretation questions for the final report

1. Why can high TPR be misleading when FPR is also above 90%?
2. Do anomaly scores rank infiltration flows above benign flows, as shown by average precision and ROC AUC?
3. How does changing the validation threshold trade missed attacks against analyst workload?
4. Why must attacks remain absent from model-fitting data in this experimental design?
5. Which raw-data problems required cleaning, and why were statistical outliers retained?
6. Would chronological or cross-day testing provide stronger evidence than a random split?
7. What CIC-IDS2018 artefacts and labelling limitations prevent deployment claims?
