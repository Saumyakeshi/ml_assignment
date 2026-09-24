# Section 2 Draft — Cyber-Attack Detection

## Dataset and task

NSL-KDD was used to classify network connections into Normal, DoS, Probe, R2L, and U2R categories. The dataset provides an official 125,973-record training partition and a 22,544-record test partition. A stratified 15% validation set was taken from the official training partition, leaving 107,077 records for fitting. The official test set was not used for preprocessing or model selection.

NSL-KDD contains 41 connection features: 38 numerical variables and three categorical variables (`protocol_type`, `service`, and `flag`). No missing values or within-partition duplicates were found. This was verified programmatically rather than assumed from the dataset documentation.

The principal limitation is severe class imbalance. The fitted training data contains 57,241 Normal and 39,038 DoS records but only 846 R2L and 44 U2R records. NSL-KDD is also an older benchmark and does not fully represent contemporary network traffic.

## Preprocessing and feature selection

Numerical columns were median-imputed if required and standardized. Categorical columns were most-frequent-imputed if required and one-hot encoded. Unknown test categories were ignored safely rather than causing execution failure. Zero-variance features were removed, followed by ANOVA F-value selection of 64 features. Every transformation was fitted on training data only and then applied unchanged to validation and test data.

## Models

The classical baseline was a 180-tree Random Forest with balanced subsample class weighting. The deep-learning comparison was a one-dimensional CNN with two convolutional blocks, batch normalization, ReLU activations, adaptive max pooling, dropout, and a five-class output layer. The CNN used square-root inverse-frequency class weights and validation macro F1 for early stopping.

The use of a CNN over a selected tabular feature vector is a modelling limitation: neighbouring feature positions do not have the same natural spatial relationship as neighbouring image pixels. Its performance must therefore be compared critically against the tree-based baseline.

## Smoke-test results

| Model | Accuracy | Macro precision | Macro recall | Macro F1 | Weighted F1 | Macro AP |
|---|---:|---:|---:|---:|---:|---:|
| Random Forest | 0.7377 | 0.7850 | 0.4738 | 0.4844 | 0.6955 | 0.6813 |
| 1D CNN — one epoch | 0.7189 | 0.6826 | 0.5263 | 0.5103 | 0.6944 | 0.6587 |

These CNN values verify the pipeline after one epoch and must be replaced after the configured full training run.

## Initial analysis

Random Forest produced the best accuracy and macro precision, while the one-epoch CNN produced better macro recall and macro F1. The CNN substantially improved Probe recall and raised U2R recall from 0.02 to 0.15, but its U2R precision was extremely low. Both models detected fewer than 7% of R2L examples. These results show why accuracy is inadequate for a security dataset dominated by Normal and DoS traffic.

The official NSL-KDD test set includes attacks absent from training. Poor R2L and U2R generalization therefore reflects both extreme scarcity and distribution shift. Potential improvements include targeted resampling within training, cost-sensitive tuning, alternative feature selection, calibrated class weights, per-class threshold analysis, and comparison with an architecture better suited to tabular data.

## Final-run placeholders

- CNN epochs completed: _replace after full run_
- Best validation macro F1: _replace after full run_
- Final test macro F1: _replace after full run_
- Final per-class recall: _replace after full run_
- Model-efficiency comparison: _replace after full run_

