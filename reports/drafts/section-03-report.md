# Section 3 Report Draft - Anomaly Detection for Cyber Threats

## Aim

The experiment investigated whether models trained only on benign network flows could identify previously unseen abnormal behaviour. Isolation Forest was compared with a dense Autoencoder using the CSE-CIC-IDS2018 flow dataset for 1 March 2018.

## Dataset and cleaning

The selected CSV contained 331,125 records and 80 source columns. It was intentionally used before model-ready cleaning. Profiling identified 97 raw duplicate rows, 1,834 native missing values, 4,000 infinite rate values, and 25 embedded repeated headers. Column and label whitespace was normalized; the embedded headers and 73 remaining duplicates were removed; infinite and malformed numeric values became missing values; timestamp features were derived; high-missingness fields could be removed; and remaining missing values were median-imputed. Twenty-four repeated headers were duplicates of one another, explaining the difference between the raw and post-header duplicate counts. Potential statistical outliers were retained because they may be the security-relevant anomalies.

After cleaning, 331,027 usable records remained: 237,987 benign and 93,040 infiltration records. To keep execution practical, the configured experiment sampled 120,000 benign and 18,000 infiltration records.

## Experimental design

Only benign flows were included in the 72,000-record training partition. The validation and test partitions each contained 24,000 benign and 9,000 infiltration records. Labels were used only to select model-specific validation thresholds and calculate final metrics. Imputation, variance filtering, and standardization were fitted on the normal training partition to prevent leakage.

## Models

Isolation Forest used 200 randomized isolation trees and up to 10,000 observations per tree. Its anomaly score was the negative tree-ensemble normality score. The Autoencoder compressed the traffic features into a 12-unit latent representation and used mean squared reconstruction error as its anomaly score. Both thresholds maximized F1 on the labelled validation partition.

## Results

| Model | TPR | FPR | Precision | F1 | Average precision | ROC AUC |
|---|---:|---:|---:|---:|---:|---:|
| Isolation Forest | 0.956 | 0.916 | 0.281 | 0.435 | 0.247 | 0.461 |
| Autoencoder (12 epochs) | 0.997 | 0.888 | 0.296 | 0.457 | 0.277 | 0.517 |

## Discussion

The high TPR values initially appear encouraging, but they result from thresholds that classify nearly all traffic as anomalous. FPR above 90% would overwhelm an analyst and makes both detectors operationally unusable in their current form. Average precision and ROC AUC near or below random ranking show that the anomaly scores do not meaningfully separate the labelled infiltration flows from benign flows.

This negative result is informative. Infiltration flows may be deliberately similar to legitimate traffic, while the selected flow-level features and simulated testbed artefacts may not provide a stable definition of normality. The validation-selected threshold behaved correctly according to its objective, but maximizing F1 at this anomaly prevalence did not impose a usable false-alert constraint. The final experiment should report threshold trade-offs rather than relying on TPR alone.

## Limitations and next experiment

The completed 12-epoch Autoencoder run reduced FPR compared with its one-epoch implementation check but remained operationally unacceptable. Additional experiments should examine a threshold constrained to a target validation FPR, a chronological or cross-day test, and sensitivity to the selected traffic features. Results cannot be interpreted as evidence of deployment readiness because CIC-IDS2018 is a controlled 2018 environment with known artefact and labelling concerns.
