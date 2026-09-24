# Assignment-Relevant Guidance Extracted from the Lectures

This document consolidates the lecture material that is directly useful for completing the COMP70049 Machine Learning in Cyber assignment. It should be read alongside the formal [assignment information](./ASSIGNMENT_INFORMATION.md).

> **Priority rule:** The assignment brief and Blackboard instructions are authoritative. Lecture guidance helps explain how to satisfy the brief, but it does not replace formal requirements.

## 1. Assignment workflow emphasized in the lectures

The recurring workflow across the lectures is:

1. Select a realistic cybersecurity dataset with enough observations, features, labels, and variation.
2. Record its source and understand its target variable.
3. Profile the data before modelling.
4. Perform exploratory data analysis and document distributions, class balance, missing values, duplicates, outliers, and feature types.
5. Clean and transform the data only where justified.
6. Split the data into training, validation, and test sets.
7. Fit preprocessing transformations on the training data only.
8. Train one classical machine-learning model and one deep-learning model.
9. Select hyperparameters with training/validation evidence rather than test-set results.
10. Evaluate both models on the same held-out test data.
11. Compare performance, efficiency, limitations, errors, and cybersecurity implications.
12. Explain the results instead of presenting metrics and figures without interpretation.

The central lecture message is that good generalization matters more than excellent training performance. A model that memorizes the training set but performs poorly on unseen data is overfitting.

Sources: [Lecture 1](../lectures/lecture-01-2026-07-12-introduction-to-machine-learning.md), [Lecture 4](../lectures/lecture-04-2026-08-02-feature-engineering.md), and [Lecture 11](../lectures/lecture-11-2026-09-20-natural-language-processing.md).

## 2. Dataset profiling and preprocessing

### What to inspect

Before training, record:

- number of rows and columns;
- feature names and data types;
- target definition and class counts;
- missing-value counts;
- duplicate records;
- unique values and feature cardinality;
- minimum, maximum, mean, median, standard deviation, and quartiles for numerical features;
- categorical-value frequencies;
- distributions, skewness, correlations, and possible outliers; and
- whether the dataset was already cleaned or engineered by its publisher.

If a dataset is already clean, show that the checks were performed and explain why further cleaning was unnecessary. Do not omit the profiling stage.

### Cleaning principles

- Choose missing-value treatment according to the feature and dataset; there is no universally correct method.
- Do not delete every row with a missing value without measuring the resulting data loss.
- Remove duplicates where they would leak repeated observations across data splits or distort training.
- Treat high-cardinality identifiers carefully. User IDs, email addresses, transaction IDs, and raw IP addresses may increase complexity without improving generalization.
- Do not automatically delete outliers. In cybersecurity, the outlier may be the attack or anomaly that the model must detect.
- Apply the same preprocessing to validation, test, and future input data.

### Scaling and categorical encoding

- Standardization uses \(z=(x-\mu)/\sigma\).
- Min-max normalization commonly maps values to the range \([0,1]\).
- Fit scalers on the training set, then reuse those fitted transformations on validation and test sets.
- Use ordinal encoding only when categories have a meaningful order.
- Use one-hot encoding for unordered categories.
- Scaling is especially important for distance-based models such as KNN and for many neural networks.

### Feature engineering

The lectures recommend converting raw security events into behaviour-oriented features. Useful categories include:

- **Raw features:** packet size, destination port, API call, timestamp, sender domain, or login status.
- **Derived features:** packets per second, failed logins in ten minutes, suspicious-link ratio, files modified per minute, or changed-extension count.
- **Interaction features:** location combined with time, port combined with packet volume, or CPU activity relative to network traffic.
- **Temporal features:** events per time window, time between events, sudden changes, and repeated activity.

Remove or combine redundant features where appropriate. Feature selection keeps useful original variables; feature extraction such as PCA creates a smaller transformed representation. Dimensionality reduction should retain meaningful security information rather than merely reducing the column count.

Sources: [Lecture 3](../lectures/lecture-03-2026-07-26-data-preprocessing-and-probability-distributions.md) and [Lecture 4](../lectures/lecture-04-2026-08-02-feature-engineering.md).

## 3. Train, validation, and test strategy

- Training data is used to fit model parameters.
- Validation data or cross-validation is used to select hyperparameters, thresholds, architectures, and stopping points.
- Test data is reserved for the final unbiased comparison.
- Use stratification for imbalanced classification data where appropriate.
- Fix random seeds where possible so results can be reproduced.
- Prevent duplicate or near-duplicate records from appearing across splits.
- Keep the split consistent when comparing classical and deep-learning models.
- Never repeatedly tune decisions against the test set.

The lectures often demonstrated an 80/20 training/test split. That is an example, not a universal rule. The assignment implementation may use a separate validation set or cross-validation when model selection requires it.

Sources: [Lecture 1](../lectures/lecture-01-2026-07-12-introduction-to-machine-learning.md), [Lecture 5](../lectures/lecture-05-2026-08-09-regression.md), [Lecture 6](../lectures/lecture-06-2026-08-16-classification-and-evaluation.md), and [Lecture 9](../lectures/lecture-09-2026-09-06-anomaly-detection-and-cnns.md).

## 4. Section 1 — Email security and phishing detection

### Lecture-derived implementation guidance

Use the following text pipeline:

\[
\text{raw email}
\rightarrow
\text{task-aware cleaning}
\rightarrow
\text{tokens or numerical representation}
\rightarrow
\text{classifier}
\rightarrow
\text{phishing probability or label}
\]

Profile the ham/phishing class counts before training. Phishing datasets are often imbalanced, so accuracy alone is not sufficient.

Text cleaning may include removing formatting artifacts, unnecessary whitespace, and selected punctuation. However, cleaning must preserve security-relevant information such as:

- URLs and suspicious domains;
- email addresses or sender-domain patterns;
- IP addresses and security identifiers;
- urgency language;
- attachment indicators;
- numbers that carry security meaning; and
- negations whose removal would change meaning.

Review any stop-word list before applying it. Stemming is fast but can create unnatural word fragments; lemmatization is usually more meaningful but costs more computation. N-grams can capture short phrases, although larger n-grams increase sparsity.

### Model connection

- **Classical model:** TF-IDF features with Logistic Regression provide a strong, interpretable baseline.
- **Deep-learning model:** an LSTM can learn ordered text patterns from token sequences and embeddings.
- **Alternative deep model:** BERT or DistilBERT provides contextual embeddings that can represent the same word differently according to its context.

The phishing lecture example compared Logistic Regression and an RNN using DistilBERT embeddings. Their performance was similar on a small dataset, while Logistic Regression trained much faster. The lecturer’s conclusion was to prefer the simplest model that meets the performance requirement; additional complexity is not automatically better.

For sequence models, document vocabulary construction, unknown and padding tokens, maximum sequence length, truncation, embedding dimensions, recurrent units, dropout, optimizer, learning rate, batch size, epochs, and early stopping.

### Evaluation focus

Report precision, recall, F1-score, ROC-AUC or the ROC curve, a confusion matrix, and preferably a precision-recall curve for imbalanced data. Discuss false negatives as missed phishing emails and false positives as legitimate emails incorrectly blocked.

Sources: [Lecture 6](../lectures/lecture-06-2026-08-16-classification-and-evaluation.md), [Lecture 10](../lectures/lecture-10-2026-09-13-word-embeddings-rnns-and-bert.md), and [Lecture 11](../lectures/lecture-11-2026-09-20-natural-language-processing.md).

## 5. Section 2 — Cyber-attack detection

### Lecture-derived implementation guidance

Network-security datasets may contain normal traffic, malicious traffic, attack categories, flow or packet statistics, missing values, noise, imbalance, and redundant features.

Candidate features discussed in the lectures include:

- packet count and packets per second;
- connection duration;
- bytes sent and received;
- average packet size;
- SYN-packet count;
- destination port;
- failed connections or login attempts;
- scanned or unique ports; and
- internal/external address indicators.

Handle missing values, one-hot encode unordered categories such as protocols, standardize numerical features where the model needs it, and investigate correlation or redundancy. Feature selection or PCA can reduce dimensionality, but any reduction must be fitted on training data and justified.

Decision Trees learn interpretable feature-based rules and provide a natural classical baseline. The assignment also permits Random Forest. For a deep model, use a CNN or RNN with an input representation that matches the structure of the network records.

### Evaluation focus

Report accuracy, class-wise precision and recall, F1-scores, a confusion matrix, and precision-recall curves. For multiclass attacks, explain which attack types are confused with each other. Do not allow a dominant normal class to hide weak attack detection.

Sources: [Lecture 2](../lectures/lecture-02-2026-07-19-python-colab-and-decision-trees.md), [Lecture 4](../lectures/lecture-04-2026-08-02-feature-engineering.md), and [Lecture 6](../lectures/lecture-06-2026-08-16-classification-and-evaluation.md).

## 6. Section 3 — Anomaly detection for cyber threats

### Lecture-derived implementation guidance

Anomaly detection learns normal behaviour and flags observations that are sufficiently unusual. It can sometimes identify previously unseen attacks because it does not require every anomaly type to be represented as a recurring labelled class.

Important cautions:

- An anomaly is not automatically an attack; it is an observation requiring investigation.
- Do not automatically remove outliers during preprocessing.
- Train the unsupervised detector on normal behaviour where the experimental design requires it.
- Use labelled anomalies for validation, threshold selection, and final evaluation when labels are available.
- Select the decision threshold on validation data, not test data.

For Gaussian anomaly detection, inspect feature distributions. Strongly skewed variables may benefit from transformations such as a logarithm. A multivariate Gaussian can represent covariance between related features.

Isolation Forest isolates sparse observations in fewer random splits than dense normal observations. Document the number of trees, sample size, contamination or score threshold, and threshold-selection method.

The assignment requires an Autoencoder as the deep-learning comparison. Train it to reconstruct normal input and classify observations with reconstruction error above a validation-selected threshold as anomalies.

### Evaluation focus

Accuracy is misleading when anomalies are rare. Report TPR/recall, FPR, precision, F1-score, a confusion matrix, a precision-recall curve, and the selected threshold. Explain the operational trade-off: a low threshold may detect more threats but create more alerts, while a high threshold may reduce false alarms but miss attacks.

Source: [Lecture 9](../lectures/lecture-09-2026-09-06-anomaly-detection-and-cnns.md).

## 7. Section 4 — Ransomware detection and prevention

### Lecture-derived implementation guidance

Convert API calls, system logs, and file activity into representations that expose ransomware behaviour. Candidate features from the lectures include:

- file modifications per minute;
- changed file extensions;
- registry modifications;
- unusual PowerShell or command-line activity;
- host-file changes;
- command-and-control connections;
- counts of API calls in a time window;
- time between calls or file events; and
- ordered API-call or behavioural sequences.

Counts in a fixed interval connect naturally to Poisson-style event analysis, while waiting times connect to exponential distributions. These distributions can support EDA and feature understanding, but the final model choice should follow the assignment brief.

For the classical model, the brief suggests SVM or Gradient Boosting. For the deep model, LSTM/RNN is appropriate because API calls and behaviour logs are ordered sequences. CNNs can also process sequences, but the implementation should clearly satisfy the formal model requirement selected for this section.

### Evaluation focus

Report precision, recall, F1-score, and a confusion matrix. Explain the security consequences of false negatives, such as undetected encryption activity, and false positives, such as legitimate administrative activity being blocked.

Sources: [Lecture 3](../lectures/lecture-03-2026-07-26-data-preprocessing-and-probability-distributions.md), [Lecture 4](../lectures/lecture-04-2026-08-02-feature-engineering.md), [Lecture 9](../lectures/lecture-09-2026-09-06-anomaly-detection-and-cnns.md), and [Lecture 10](../lectures/lecture-10-2026-09-13-word-embeddings-rnns-and-bert.md).

## 8. Evaluation and interpretation

### Classification metrics

- **Accuracy:** proportion of all predictions that are correct.
- **Precision:** of the observations predicted as attacks, how many are actual attacks?
- **Recall/TPR:** of all actual attacks, how many were detected?
- **F1-score:** harmonic mean of precision and recall.
- **FPR:** proportion of actual normal observations incorrectly flagged as attacks.
- **ROC curve:** TPR against FPR across decision thresholds.
- **Precision-recall curve:** precision against recall across thresholds; especially informative for rare attacks.
- **Confusion matrix:** counts of true positives, true negatives, false positives, and false negatives.

Accuracy alone is insufficient for imbalanced security data. False negatives can be particularly serious because they represent missed threats, but false positives also matter because excessive alerts or blocked legitimate activity can make a security system unusable.

### Model comparison

Compare models using more than the largest metric value. Discuss:

- generalization to unseen data;
- per-class and minority-class performance;
- types of mistakes made;
- training and inference time;
- memory and compute requirements;
- interpretability;
- sensitivity to preprocessing and hyperparameters;
- evidence of underfitting or overfitting; and
- likely performance in the intended cybersecurity setting.

Synthetic or unusually simple data can produce perfect or near-perfect scores. Treat this as a reason to inspect leakage, duplicates, class separability, and realism rather than as automatic evidence of a production-ready model.

Sources: [Lecture 2](../lectures/lecture-02-2026-07-19-python-colab-and-decision-trees.md), [Lecture 6](../lectures/lecture-06-2026-08-16-classification-and-evaluation.md), and [Lecture 9](../lectures/lecture-09-2026-09-06-anomaly-detection-and-cnns.md).

## 9. Deep-learning training evidence

For every deep-learning experiment, save and discuss training and validation loss curves and, where appropriate, accuracy or task-specific metric curves.

Typical interpretations are:

- both losses decreasing and stabilizing: healthy learning;
- both losses remaining high: underfitting or an optimization/data problem;
- training loss decreasing while validation loss rises: overfitting; and
- loss oscillating: possible learning-rate, scaling, data, or optimizer problem.

Possible responses to overfitting include early stopping, dropout, regularization, a smaller model, more representative data, augmentation where meaningful, and better data separation. Save final model weights and all hyperparameters needed to reproduce the run.

Sources: [Lecture 7](../lectures/lecture-07-2026-08-23-artificial-neural-networks.md) and [Lecture 9](../lectures/lecture-09-2026-09-06-anomaly-detection-and-cnns.md).

## 10. Report and submission guidance

The lecturer emphasized implementation quality and explanation, not only final numeric scores. For every section, the report should explain:

- dataset source, target, size, features, class distribution, and limitations;
- profiling and cleaning checks, including cases where no change was needed;
- feature engineering and preprocessing decisions;
- split strategy and leakage prevention;
- model choice, architecture, important hyperparameters, and training process;
- metrics and visualizations;
- why one model performed differently from the other;
- important errors and cybersecurity consequences;
- limitations and realistic improvements; and
- how the work can be reproduced.

Use the appendix for code, dataset links, notebook links, and supporting detail where permitted. A confusion matrix or metric table must be interpreted; do not insert it without explaining its meaning.

### Word-count conflict

Lecture 11 records a suggestion that a longer 5,000–6,000-word explanation might be acceptable. The formal assignment brief in this repository states **a maximum of 3,000 words**. Follow the formal 3,000-word limit unless Blackboard or the module leader provides an explicit authoritative amendment.

Source: [Lecture 11](../lectures/lecture-11-2026-09-20-natural-language-processing.md).

## 11. Recommended evidence checklist for each section

- [ ] Dataset source and access link recorded.
- [ ] Dataset dimensions, schema, target, and class distribution shown.
- [ ] Missing values and duplicates checked.
- [ ] Numerical distributions, skewness, and outliers reviewed.
- [ ] Cleaning and feature decisions justified.
- [ ] Train/validation/test strategy documented.
- [ ] Preprocessors fitted only on training data.
- [ ] One classical and one deep-learning model implemented.
- [ ] Hyperparameters and random seeds recorded.
- [ ] Required metrics and confusion matrix saved.
- [ ] Threshold curves or training curves saved where relevant.
- [ ] Errors, limitations, security implications, and improvements discussed.
- [ ] Commands or notebook instructions tested from a clean environment.

## 12. Lecture-to-assignment map

| Lecture | Most relevant assignment contribution |
|---|---|
| [Lecture 1](../lectures/lecture-01-2026-07-12-introduction-to-machine-learning.md) | Train/test separation, generalization, underfitting, and overfitting |
| [Lecture 2](../lectures/lecture-02-2026-07-19-python-colab-and-decision-trees.md) | Colab workflow, Decision Trees, confusion matrix, and classification metrics |
| [Lecture 3](../lectures/lecture-03-2026-07-26-data-preprocessing-and-probability-distributions.md) | Profiling, cleaning, distributions, outliers, splitting, and event distributions |
| [Lecture 4](../lectures/lecture-04-2026-08-02-feature-engineering.md) | Security feature engineering, scaling, encoding, PCA, and assignment workflow |
| [Lecture 5](../lectures/lecture-05-2026-08-09-regression.md) | Reproducible splitting, profiling evidence, and matching metrics to task type |
| [Lecture 6](../lectures/lecture-06-2026-08-16-classification-and-evaluation.md) | Logistic Regression, cross-validation, scaling, imbalance, and phishing evaluation |
| [Lecture 7](../lectures/lecture-07-2026-08-23-artificial-neural-networks.md) | Neural-network training, learning curves, convergence, and regularization |
| [Lecture 9](../lectures/lecture-09-2026-09-06-anomaly-detection-and-cnns.md) | Isolation Forest, anomaly thresholds, precision-recall evaluation, CNNs, and regularization |
| [Lecture 10](../lectures/lecture-10-2026-09-13-word-embeddings-rnns-and-bert.md) | Embeddings, RNN/LSTM sequence modelling, BERT, and phishing classification |
| [Lecture 11](../lectures/lecture-11-2026-09-20-natural-language-processing.md) | Text preprocessing, assignment explanation, appendices, and result interpretation |
