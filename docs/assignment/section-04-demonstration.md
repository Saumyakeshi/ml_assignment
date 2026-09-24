# Section 4 Demonstration - Ransomware Detection and Prevention

## Assignment alignment

This experiment detects ransomware from ordered Windows API-call behavior using:

- **Dataset:** MalwareAPI-2026 API-call sequences;
- **Classical model:** class-balanced linear Support Vector Machine;
- **Deep-learning model:** class-weighted LSTM;
- **Evaluation:** precision, recall, F1-score, confusion matrix, ROC AUC, and
  average precision.

See the [dataset record](../research/ransomware-api-sequence-dataset.md) for
provenance, checksums, profiling, and limitations.

## Data preparation

The notebook:

1. downloads the published 68.4 MB CSV from Zenodo;
2. retains only `Goodware` and `ransomware` records;
3. checks missing values and exact duplicate sequences;
4. removes 744 duplicate sequences before splitting;
5. creates one stratified train/validation/test split shared by both models; and
6. builds all vocabularies and representations from training data only.

After deduplication, only 39 of the 630 sequences are ransomware. This is an
important result rather than a reason to keep duplicates: identical traces in
multiple splits would create leakage.

## Models

### Linear SVM

API calls are represented using TF-IDF unigram and bigram features. A linear SVM
uses balanced class weights to give the small ransomware class more influence.

### LSTM

The vocabulary is built from training calls only. Each 100-call sequence is
integer encoded and processed by a 64-dimensional embedding and 64-unit LSTM,
followed by dropout and a binary output. Weighted binary cross-entropy addresses
the class imbalance, and the state with the best validation F1 is retained.

## Validated results

| Model | Accuracy | Precision | Recall | F1 | ROC AUC | Average precision |
|---|---:|---:|---:|---:|---:|---:|
| Linear SVM | 0.9579 | 1.0000 | 0.3333 | 0.5000 | 0.9270 | 0.6110 |
| LSTM | 0.8947 | 0.2500 | 0.3333 | 0.2857 | 0.8296 | 0.4546 |

Both models detected two of the six ransomware sequences in the test set. The
SVM generated no false positives, while the LSTM generated six. The SVM is the
stronger result for this small dataset, but the test ransomware count is too low
for a stable deployment claim.

## Security interpretation and prevention

- A false negative is ransomware behavior that may continue to file encryption,
  exfiltration, or destruction.
- A false positive may quarantine legitimate software and interrupt normal work.
- Model alerts should trigger containment and investigation rather than serving
  as the only preventive control.
- Practical prevention also requires offline or immutable backups, tested
  recovery procedures, least privilege, application allow-listing, patching,
  endpoint monitoring, network segmentation, and user awareness.

## Run the notebook

Open
[`notebooks/04_ransomware/section-04-ransomware-detection.ipynb`](../../notebooks/04_ransomware/section-04-ransomware-detection.ipynb)
locally or through the VS Code Colab extension and run it from top to bottom.
Set `LSTM_EPOCHS = 1` near the top for a quick pipeline check or retain 15 for
the validated experiment.

## Outputs

```text
data/processed/section_04/
|-- data-profile.json
`-- split-manifest.csv

models/section_04/
|-- svm.joblib
`-- lstm.pt

reports/section_04/
|-- model-comparison.csv
|-- run-summary.json
|-- figures/
|   |-- class-distribution.png
|   |-- svm-evaluation.png
|   |-- lstm-evaluation.png
|   `-- lstm-training-history.png
`-- metrics/
    |-- svm.json
    `-- lstm.json
```
