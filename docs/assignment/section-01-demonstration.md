# Section 1 Demonstration - Email Security

## Objective

Demonstrate the complete Section 1 workflow using raw email text:

```text
Official corpus
    -> MIME parsing and deduplication
    -> stratified train/validation/test split
    -> TF-IDF + Logistic Regression
    -> token vocabulary + LSTM
    -> shared test-set evaluation
    -> metrics, plots, model comparison, and report evidence
```

## Scope

The assignment brief recommends the SpamAssassin Dataset. Its labels are spam and ham rather than phishing and legitimate email. This implementation is therefore a defensible demonstration of the required flow, but the limitation must be stated clearly in the report.

See [the dataset note](../research/spamassassin-public-corpus.md) for provenance and limitations.

## Implemented models

### Classic machine learning

- Text representation: TF-IDF unigrams and bigrams
- Classifier: Logistic Regression
- Imbalance handling: balanced class weights
- Leakage protection: the vectorizer is fitted only on the training set through a scikit-learn pipeline

### Deep learning

- Text representation: learned token embeddings
- Network: single-layer LSTM followed by dropout and a binary output layer
- Loss: weighted binary cross-entropy with logits
- Training control: validation F1 and early stopping
- Leakage protection: the vocabulary is constructed only from training messages

## Evaluation

Both models use the same held-out test set and a fixed threshold of 0.5. The run produces:

- Accuracy
- Precision
- Recall
- F1-score
- ROC AUC
- Average precision
- Classification report
- Confusion matrix
- ROC curve
- Precision-recall curve
- LSTM training history

## Run instructions

The complete narrative and executable workflow is available in the
[Section 1 Colab notebook](../../notebooks/01_phishing/section-01-email-security.ipynb).

From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[deep,dev]"
python scripts/download_spamassassin.py
python scripts/run_section_01.py
pytest
```

For a fast classic-model-only check:

```powershell
python scripts/run_section_01.py --skip-lstm
```

For a one-epoch LSTM smoke test:

```powershell
python scripts/run_section_01.py --epochs 1
```

## Outputs

| Location | Contents |
|---|---|
| `data/processed/section_01/split_manifest.csv` | Reproducible split membership without duplicated message text |
| `models/section_01/` | Saved TF-IDF/Logistic Regression pipeline and LSTM checkpoint |
| `reports/section_01/model-comparison.csv` | Side-by-side test metrics |
| `reports/section_01/metrics/` | Detailed metrics and classification reports |
| `reports/section_01/figures/` | Confusion matrices and ROC/precision-recall curves |
| `reports/section_01/run-summary.json` | Dataset and split summary |

## Interpretation rules

- Prefer precision, recall, F1, ROC AUC, and average precision over accuracy alone because the classes are imbalanced.
- Discuss false positives as legitimate messages that would be blocked.
- Discuss false negatives as malicious or unwanted messages that would reach the user.
- Compare the models on identical test records.
- Do not claim that a higher score proves one model is universally superior; consider runtime, explainability, data requirements, and corpus age.
