# Section 3 Demonstration - Anomaly Detection for Cyber Threats

## Assignment alignment

This experiment implements the Section 3 requirements with:

- **Dataset:** UNSW-NB15 network-flow data;
- **Classical model:** Isolation Forest;
- **Deep-learning model:** denoising Autoencoder;
- **Training:** benign traffic only;
- **Threshold selection:** minimum validation false-positive rate while detecting at least 60% of validation attacks; and
- **Evaluation:** TPR, FPR, precision-recall curves, ROC curves, confusion matrices, F1, average precision, and ROC AUC.

## Dataset and download

UNSW-NB15 combines normal activity with nine attack families: Analysis, Backdoor, DoS, Exploits, Fuzzers, Generic, Reconnaissance, Shellcode, and Worms. The notebook downloads the published 175,341-record development file and 82,332-record test file from public mirrors, records SHA-256 hashes, and links the official UNSW dataset page in the run summary.

The university's current SharePoint download requires authentication, so the notebook uses two public download mirrors with automatic fallback. The source release remains the UNSW-NB15 standard training and testing CSV files.

## Cleaning and preprocessing

The pipeline:

1. normalizes column names, labels, and attack-family names;
2. converts numerical fields safely and replaces infinite values with missing values;
3. removes exact duplicate feature records inside each partition;
4. removes exact test representations also present in development data;
5. fits median imputation only on benign training records;
6. applies a signed `log1p` transform to reduce extreme numerical skew;
7. applies robust scaling using the 5th and 95th percentiles;
8. one-hot encodes protocol, service, and connection state; and
9. removes zero-variance transformed features.

Cleaning removed 74,301 duplicate development records, 28,386 duplicate test records, and 1,302 overlapping test representations. The model-ready experiment contained 58 transformed features.

## Leakage-safe anomaly experiment

- **Training:** 15,000 benign development records; attack labels are not used for model fitting.
- **Validation:** 12,000 benign and 12,000 attack records; labels tune hyperparameters and thresholds.
- **Test:** 52,644 records from the held-out standard test file after cleaning, containing 33,662 benign and 18,982 attack records.

All preprocessing is fitted on benign training data. The test partition is used only after model and threshold selection.

## Models

### Isolation Forest

Six settings were compared on validation data. Each used 300 trees, while `max_samples` and `max_features` varied. Average precision was the primary selection measure because the classes are imbalanced. The selected model used all 15,000 benign training records per tree and all transformed features.

### Denoising Autoencoder

The Autoencoder uses a `128 -> 64 -> 16 -> 64 -> 128` hidden structure around the input and output layers. Small Gaussian noise is added during training, encouraging the network to learn stable benign patterns instead of copying individual records. Layer normalization, dropout, AdamW, gradient clipping, Smooth L1 loss, and benign-validation early stopping improve stability. Feature-level reconstruction errors are normalized using benign validation errors before the final anomaly score is calculated.

## Validated results

| Model | Accuracy | TPR | FPR | Precision | F1 | Average precision | ROC AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| Isolation Forest | 0.729 | 0.615 | 0.206 | 0.627 | 0.621 | 0.672 | 0.763 |
| Denoising Autoencoder | 0.805 | 0.709 | 0.142 | 0.739 | 0.724 | 0.796 | 0.876 |

Both models meet the requirement to categorize more than half of attacks as attacks. Isolation Forest detects 61.5% and the Autoencoder detects 70.9%. The Autoencoder remains stronger, with ROC AUC of 0.876 and average precision of 0.796. Increasing recall requires more alerts: test FPR is 20.6% for Isolation Forest and 14.2% for the Autoencoder.

Per-attack recall is uneven. The Autoencoder detects all Analysis attacks and approximately 94.9% of Generic, 85.9% of Exploits, 83.7% of Worms, and 77.7% of DoS records. It remains weak on Shellcode, Fuzzers, and Reconnaissance, so the overall score must not be interpreted as equal protection against every threat.

## Run the notebook

Open [`notebooks/03_anomaly_detection/section-03-anomaly-detection.ipynb`](../../notebooks/03_anomaly_detection/section-03-anomaly-detection.ipynb) locally or in Colab and run it from top to bottom. It downloads and validates the data, performs cleaning, tunes and evaluates both models, saves model artifacts, and creates `section_03_results.zip` in Colab.

## Interpretation questions

1. Why is average precision important when attack prevalence is unequal?
2. Why does targeting higher attack recall increase the number of false alerts?
3. Why must attacks remain absent from model-fitting data in an unsupervised experiment?
4. Which attack families are detected reliably, and which remain difficult?
5. Why do duplicate and overlapping representations create misleading evaluation results?
6. What limitations prevent deployment claims from a controlled benchmark dataset?
