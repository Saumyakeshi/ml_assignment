# Section 2 Demonstration — Cyber-Attack Detection

This walkthrough implements the complete Section 2 flow using the assignment-recommended NSL-KDD dataset.

Interactive version: [Section 2 Colab notebook](../../notebooks/02_intrusion_detection/section-02-intrusion-detection.ipynb).

## Objective

Classify each network connection into one of five categories:

- `normal`;
- `dos`;
- `probe`;
- `r2l`; or
- `u2r`.

The comparison uses:

- **Classical ML:** class-balanced Random Forest;
- **Deep learning:** class-weighted one-dimensional CNN.

## Dataset

The official UNB host no longer provides the original files, but explicitly permits redistribution and mirroring. The project downloads a validated public mirror of the original attack-labelled TXT files.

See [the NSL-KDD dataset record](../research/nsl-kdd-dataset.md) for provenance, hashes, limitations, and citation information.

## Setup

Open the [Section 2 notebook](../../notebooks/02_intrusion_detection/section-02-intrusion-detection.ipynb)
locally or through the VS Code Colab extension and run it from top to bottom.
The notebook installs its dependencies and defines every required function
inline. It does not need a repository clone, package installation, or uploaded
source files.

## Download and validate the data

The notebook's data cell downloads the official train/test files and checks:

- 125,973 training records;
- 22,544 test records;
- 43 columns per record; and
- fixed SHA-256 checksums.

## Preprocessing

The pipeline is fitted only on the training partition:

1. Split 15% of `KDDTrain+` into a stratified validation set.
2. Check missing values and duplicates.
3. Median-impute numerical columns if needed.
4. Standardize numerical columns.
5. Most-frequent-impute categorical columns if needed.
6. One-hot encode `protocol_type`, `service`, and `flag` while tolerating unseen test categories.
7. Remove zero-variance features.
8. Retain 64 features using ANOVA F-value feature selection.
9. Apply the fitted transformations unchanged to validation and official test records.

No missing values or within-partition duplicates were found, but the checks are saved as evidence in `data/processed/section_02/data-profile.json`.

## Class imbalance

The fitted training partition contains:

| Class | Records |
|---|---:|
| Normal | 57,241 |
| DoS | 39,038 |
| Probe | 9,908 |
| R2L | 846 |
| U2R | 44 |

The Random Forest uses balanced subsample weights. The CNN uses square-root inverse-frequency class weights to improve minority-class learning without applying the extremely large full inverse-frequency weights directly.

## Run the models

Run the notebook cells through the Random Forest section for the classical
baseline only. Set `CNN_EPOCHS_OVERRIDE = 1` in the configuration cell for a
quick CNN pipeline check, or leave it as `None` for the configured eight-epoch
experiment with early stopping. A one-epoch result is a pipeline check, not the
final deep-learning experiment.

## Current verified smoke results

| Model | Accuracy | Macro precision | Macro recall | Macro F1 | Weighted F1 | Macro AP |
|---|---:|---:|---:|---:|---:|---:|
| Random Forest | 0.7377 | 0.7850 | 0.4738 | 0.4844 | 0.6955 | 0.6813 |
| 1D CNN — one epoch | 0.7189 | 0.6826 | 0.5263 | 0.5103 | 0.6944 | 0.6587 |

### Initial interpretation

- Random Forest has higher accuracy and macro precision.
- The one-epoch CNN has higher macro recall and macro F1.
- Both models perform well on normal, DoS, and Probe relative to R2L and U2R.
- Random Forest U2R recall is only 0.02; the one-epoch CNN increases it to 0.15 but with very low U2R precision.
- R2L recall remains below 0.07 for both models.
- Accuracy hides these failures because normal and DoS records dominate the dataset.
- The official test set contains previously unseen attack types, so it is substantially harder than a random split from one combined table.

Do not present the one-epoch CNN as the final model. Run the configured experiment, inspect the validation curves, and update the report with the final saved metrics.

## Outputs

```text
data/processed/section_02/
|-- data-profile.json
`-- selected-features.json

models/section_02/
|-- random-forest.joblib
`-- cnn.pt

reports/section_02/
|-- model-comparison.csv
|-- run-summary.json
|-- figures/
|   |-- training-class-distribution.png
|   |-- random-forest-confusion-matrix.png
|   |-- random-forest-precision-recall-curves.png
|   |-- 1d-cnn-confusion-matrix.png
|   |-- 1d-cnn-precision-recall-curves.png
|   `-- cnn-training-history.png
`-- metrics/
    |-- random-forest.json
    `-- cnn.json
```

## Report questions

The Section 2 analysis should answer:

1. Why is accuracy misleading for these five classes?
2. Which attack classes are confused with normal traffic?
3. How do unseen test attacks affect generalization?
4. Did class weighting meaningfully improve R2L and U2R recall?
5. Does the CNN justify its additional computation compared with Random Forest?
6. What contemporary-data limitations prevent deployment claims?
