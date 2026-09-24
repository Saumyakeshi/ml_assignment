# CSE-CIC-IDS2018 for Section 3

## Selection

Section 3 uses the CSE-CIC-IDS2018 CICFlowMeter CSV for Thursday 1 March 2018:

```text
Thursday-01-03-2018_TrafficForML_CICFlowMeter.csv
```

The assignment explicitly suggests CIC-IDS2018 for anomaly detection. The selected day contains benign and infiltration traffic, making it suitable for a normal-only training experiment with labelled anomalies reserved for validation and testing.

Authoritative sources:

- [Canadian Institute for Cybersecurity dataset page](https://www.unb.ca/cic/datasets/ids-2018.html)
- [Official public AWS dataset bucket](https://registry.opendata.aws/cse-cic-ids2018/)
- [Dataset-quality corrections and limitations](https://intrusion-detection.distrinet-research.be/CNS2022/CSECICIDS2018.html)

The official page permits redistribution and mirroring when the dataset is cited and linked. Raw files remain ignored by Git because this selected CSV is approximately 103 MiB.

## Why this is an unclean dataset

The downloaded CSV is not treated as model-ready. The initial audit found:

| Issue | Observed quantity |
|---|---:|
| Raw records | 331,125 |
| Raw columns | 80 |
| Exact duplicate rows in the raw file | 97 |
| Native missing values | 1,834 |
| Infinite numeric values | 4,000 |
| Embedded repeated header rows | 25 |

The pipeline strips column whitespace, removes 25 embedded headers and then 73 remaining exact duplicate rows, replaces positive and negative infinity with missing values, coerces malformed numeric values if present, derives hour and weekday from timestamps, and median-imputes remaining missing values. The difference between 97 raw duplicates and 73 duplicate removals occurs because 24 of the repeated headers are themselves duplicates. It deliberately does **not** discard statistical outliers because unusual flows are the observations the models must detect.

## Experimental use of labels

The raw `Label` column is converted to a binary evaluation label:

- `0`: benign;
- `1`: anomaly (`Infilteration` in the original file).

Labels are not supplied to Isolation Forest or the Autoencoder during fitting. Both models train only on benign traffic. Labels are used to select an anomaly-score threshold on the validation partition and to calculate final test metrics.

## Sampling and split policy

The reproducible default experiment samples at most 120,000 benign records and 18,000 anomaly records for practical execution in Colab and on a local CPU. Benign records are divided into 60% training, 20% validation, and 20% test. Anomalies are divided equally between validation and test. Consequently:

- training contains 72,000 benign records and no labelled attacks;
- validation contains 24,000 benign and 9,000 anomaly records;
- test contains 24,000 benign and 9,000 anomaly records.

This split supports threshold selection without examining the test results. It is a random benchmark split rather than a chronological deployment simulation.

## Limitations

- It represents a controlled 2018 cloud testbed, not current production traffic.
- The selected day tests infiltration only; it does not represent every attack family.
- Published research has identified labelling and artefact issues in CIC intrusion datasets.
- Extracted flow features are not the same as completely raw packet captures.
- A random split may share environmental conditions across partitions.
- An anomaly is not automatically malicious, and operational investigation would still be required.
