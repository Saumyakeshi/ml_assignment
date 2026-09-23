# Section 1 - Email Security and Phishing Detection

## 1. Aim

This experiment compares a classic machine-learning text classifier with a deep-learning sequence model for distinguishing unwanted email from legitimate email.

## 2. Dataset

Describe the Apache SpamAssassin public corpus, class counts after parsing and deduplication, the train/validation/test split, and the spam-versus-phishing limitation documented in `docs/research/spamassassin-public-corpus.md`.

## 3. Preprocessing

Explain:

- MIME parsing and attachment exclusion
- Subject and body concatenation
- Lowercasing and whitespace normalization
- URL and email-address placeholder tokens
- Punctuation removal
- Exact-duplicate removal before splitting
- Training-only fitting of vectorizers and vocabularies

## 4. Models

### 4.1 TF-IDF and Logistic Regression

Document the n-gram range, feature limit, document-frequency thresholds, balanced class weights, and why this combination is a strong interpretable baseline.

### 4.2 LSTM

Document the vocabulary limit, sequence length, embedding dimension, hidden dimension, dropout, class-weighted loss, optimizer, early stopping, and device used.

## 5. Results

Insert the generated `reports/section_01/model-comparison.csv` table and figures. Report accuracy, precision, recall, F1, ROC AUC, and average precision for both models.

## 6. Discussion

Discuss:

- Which model detects more positive messages
- Which model generates fewer false alarms
- Whether the LSTM improvement, if any, justifies its additional complexity
- How imbalance affects accuracy
- Sources of leakage or over-optimistic performance
- Why spam performance does not directly establish phishing performance
- Expected degradation on modern or organization-specific email

## 7. Cybersecurity implications

Explain the consequences of false negatives and false positives, the need for threshold tuning, human review, continuous retraining, adversarial adaptation, privacy protection, and monitoring for distribution shift.

## 8. Improvements

Potential extensions include a phishing-specific corpus, near-duplicate grouping, sender/header features, pretrained embeddings, BERT, probability calibration, threshold selection based on security costs, and external validation on a newer dataset.

