# Lecture–notebook alignment review

Updated 24 September 2026. Scope: all ten available lecture-note files, the four revised notebooks, the extracted assignment brief, dataset records, saved profiles, metric JSON files, and report drafts. Lecture 8 has no usable notes in the [lecture index](../lectures/README.md), so its content cannot be assessed.

## Overall finding

The revised notebooks follow the taught workflow and address the earlier code-level alignment gaps. They compare an appropriate classical model with a neural model, keep learned preprocessing inside the training boundary, use validation data for checkpoint or threshold selection, report security-relevant metrics, and retain run-provenance information.

The notebooks are now **source-complete but not result-complete**. Their stale stored outputs were cleared because the revised preprocessing, conflict policy and threshold policies change the experiments. Existing files under `reports/section_NN` are legacy results and must not be attributed to the revised code. A fresh top-to-bottom run of every notebook is required before final submission, followed by export of each section's matching artifacts as one set.

The formal [assignment brief](ASSIGNMENT_INFORMATION.md) limits the report to **3,000 words**. The expanded 4,000-word report remains a drafting aid and must be shortened or supported by an authoritative amendment before submission.

## Coverage of every available lecture

| Lecture | Relevant teaching | Current notebook alignment and remaining limitation |
|---|---|---|
| [1: Introduction](../lectures/lecture-01-2026-07-12-introduction-to-machine-learning.md) | Learning from data, supervised/unsupervised learning, generalization | Sections 1, 2 and 4 are supervised; Section 3 fits normal-only models. Holdouts are separated, but the datasets do not establish production or zero-day performance. |
| [2: Python, Colab and trees](../lectures/lecture-02-2026-07-19-python-colab-and-decision-trees.md) | Notebook workflow, features versus targets, trees, confusion matrices, misleading accuracy | All four notebooks are self-contained and report confusion matrices. Local/Colab path handling is guarded, and Section 2 supplies the tree ensemble. |
| [3: Preprocessing and distributions](../lectures/lecture-03-2026-07-26-data-preprocessing-and-probability-distributions.md) | Inspect types, missingness, duplicates, distributions, skewness and outliers; split appropriately | The notebooks now include class/length or input-feature plots, missingness and duplicate checks, skewness summaries, correlation summaries where applicable, and representation-overlap checks. Learned and data-dependent preprocessing is fitted on training data. |
| [4: Feature engineering](../lectures/lecture-04-2026-08-02-feature-engineering.md) | Encoding, scaling, feature selection versus extraction, meaningful behaviour features | TF-IDF, stemming, embeddings, one-hot encoding, ANOVA selection, missingness indicators and temporal features are used. Section 3 now treats protocol and destination-port role semantically rather than as continuous magnitudes. |
| [5: Regression](../lectures/lecture-05-2026-08-09-regression.md) | Reproducible splitting, exploration even for clean data, model complexity | Seeds, split manifests, dataset hashes and environment versions support reproducibility. Regression prediction metrics such as R² are not required for these classification tasks; autoencoder reconstruction MSE has a different role. |
| [6: Classification and evaluation](../lectures/lecture-06-2026-08-16-classification-and-evaluation.md) | Imbalance, precision/recall/F1, validation, train-only preprocessing, threshold trade-offs | Class weighting, macro metrics, PR/ROC analysis and training-only transforms align. Sections 1 and 4 select thresholds under a 1% validation false-alert budget; Section 3 uses a 5% validation false-positive-rate budget. |
| [7: Neural networks](../lectures/lecture-07-2026-08-23-artificial-neural-networks.md) | Loss, backpropagation, activations, dropout, training versus validation curves | All neural implementations use these mechanisms and record training and validation losses. They restore a selected checkpoint after a fixed epoch budget; this is checkpoint selection, not early stopping. |
| [9: Anomalies and CNNs](../lectures/lecture-09-2026-09-06-anomaly-detection-and-cnns.md) | Normality versus maliciousness, isolation, thresholding, convolutions | Section 3 follows normal-only fitting and validation threshold selection with an explicit alert budget. Section 2 still convolves over selected feature columns, which have no guaranteed spatial locality. |
| [10: Embeddings, RNNs and BERT](../lectures/lecture-10-2026-09-13-word-embeddings-rnns-and-bert.md) | Token identity, learned embeddings, ordered context, recurrent memory, cost | Sections 1 and 4 use training-only vocabularies, learned embeddings and LSTMs. Pretrained embeddings and BERT remain alternatives rather than compulsory additions. Runtime and memory comparisons are not measured. |
| [11: NLP and assignment guidance](../lectures/lecture-11-2026-09-20-natural-language-processing.md) | Domain-aware cleaning, stopwords, stemming/lemmatization, explain decisions and failures | Section 1 now stems both branches, removes security-reviewed English stopwords in TF-IDF, retains stopwords for LSTM sequence context, preserves negation, and records parsing/normalization diagnostics. |

PCA, BERT, formal normality tests, a particular 80/20 split, and every candidate feature mentioned in lectures are not individually compulsory. The important distinction is between an explicit brief requirement, a teaching recommendation, and an illustrative alternative. The fixed 70/15/15 splits remain defensible, while seed 42 is a reproducibility convention rather than a course requirement.

## Section-by-section review

### 1. Email security

[Notebook](../../notebooks/01_phishing/section-01-email-security.ipynb): MIME parsing with malformed-charset accounting; plain-text preference with HTML fallback; attachment exclusion; Unicode-preserving normalization; URL/address markers; stemming; normalized-text conflict exclusion and deduplication; stratified shared splits; training-only TF-IDF/vocabulary; weighted Logistic Regression and LSTM; validation-selected thresholds; confusion, ROC and PR plots.

The earlier text-processing and evidence gaps are resolved in source. The notebook records message lengths, charset fallbacks, replacement characters, empty normalized messages, conflicting labels and normalized-representation overlap. Its stopword policy deliberately preserves negation and applies generic stopword removal only to TF-IDF; the LSTM retains function words because ordered context can be meaningful.

Remaining limitations:

- SpamAssassin supplies spam/ham labels, not phishing/legitimate labels. Every result claim must retain that distinction.
- Replacing addresses and URLs with markers removes their identity and structure. This is a deliberate generalisation trade-off rather than a claim that those details are unimportant.
- Exact normalized-text separation does not establish independence between near-duplicate campaigns.
- The revised experiment has not yet been executed, so no current result may be quoted until a fresh, internally consistent artifact set is exported.

### 2. Intrusion detection

[Notebook](../../notebooks/02_intrusion_detection/section-02-intrusion-detection.ipynb): official NSL-KDD test partition; training-only scaling, one-hot encoding, variance filtering and ANOVA selection; shared 64-feature representation; Random Forest and 1D CNN; five-class metrics; per-class PR curves; predictor-only duplicate and cross-partition overlap checks; input distributions, skewness and correlation summaries.

The notebook correctly makes no imputation claim: the recorded source profiles contain no missing values. The fitted preprocessing pipeline is saved independently for the CNN as well as packaged with the Random Forest. CNN history now includes validation loss, validation macro F1 and the chosen epoch.

Remaining limitations:

- The CNN convolves across selected feature columns, not successive packets. Adjacent columns have no guaranteed spatial relationship, so it must not be described as temporal traffic modelling.
- Rare-class recall remains essential. New results must report R2L and U2R counts and recall rather than relying on accuracy or weighted averages.
- Output scores are not probability-calibrated, and no systematic hyperparameter-search study is implemented.
- NSL-KDD is an old benchmark; its official holdout does not establish performance on current network traffic.

### 3. Anomaly detection

[Notebook](../../notebooks/03_anomaly_detection/section-03-anomaly-detection.ipynb): genuine dirty CIC-IDS2018 flow CSV; header and duplicate cleaning; infinity replacement; benign-only training; training-fitted missingness selection, imputation indicators, semantic protocol/port encoding, variance filtering and scaling; Isolation Forest and Autoencoder; validation false-alert budget; TPR/FPR/PR evaluation.

The earlier preprocessing leakage is removed: the 40% missingness decision is now fitted only on benign training records. Protocol is categorical, and destination ports are represented by common-service or IANA-range categories. Input distributions, skewness, correlations and exact representation overlap are recorded. The fitted preprocessor is saved as a standalone artifact for both model families.

Labels select benign training and validation pools, define evaluation sets and select thresholds. They do not enter the Isolation Forest fit or Autoencoder reconstruction loss.

Remaining limitations:

- A random split from one captured day does not establish future-day, unseen-family or zero-day performance.
- Exact row/representation checks cannot prove that related flows or attacks are independent across partitions.
- The 5% validation false-positive-rate budget is an explicit experimental choice, not a demonstrated production requirement. Its operational suitability must be justified in the report.
- The revised experiment must be rerun before claiming that its test false-positive rate satisfies the validation budget; the budget constrains validation, not the unseen test distribution.

### 4. Ransomware

[Notebook](../../notebooks/04_ransomware/section-04-ransomware-detection.ipynb): ordered API traces; Goodware/ransomware filtering; explicit exclusion of all conflicting-label sequences; exact sequence deduplication; shared stratified splits; training-only API TF-IDF and LSTM vocabulary; balanced linear SVM and weighted LSTM; validation-selected thresholds.

The conflict policy is now deterministic and auditable. Both ambiguous sequences and all eight associated rows are excluded before deduplication. On the audited local dataset this leaves 590 goodware and 38 ransomware sequences. The conflict audit stores sequence hashes and per-label row counts. The SVM is evaluated with raw decision margins, without presenting a sigmoid transformation as calibrated probability.

Remaining limitations:

- With only 38 ransomware sequences, validation and test contain roughly six positives each. One prediction changes recall by about 16.7 percentage points, so counts and uncertainty must accompany percentage metrics.
- The source lacks a reliable malware-family grouping field. Exact sequence deduplication therefore does not establish family-independent generalisation.
- Raw SVM margins and LSTM sigmoid outputs are ranking scores, not calibrated deployment probabilities.
- The notebook classifies completed traces. It does not implement live blocking, rollback, time-to-detection evaluation or proof that encryption was prevented.

## Result provenance and rerun requirement

The revised notebooks record dataset hashes, seeds, package versions, platform details, split membership, numerical training histories, chosen epochs and selected features. Colab exports include processed profiles/manifests as well as reports and models. Each section's exported files must come from one uninterrupted top-to-bottom execution.

All notebook outputs were intentionally cleared. The existing JSON, CSV, figures and model files elsewhere in the repository predate the revised pipelines. In particular, they reflect the previous Section 1 preprocessing, Section 3 feature boundary/threshold policy, and Section 4 conflict handling. They are useful only as legacy records and must not be mixed with a new training history or presented as reproduction of the revised notebooks.

Before submission:

1. Run each notebook from a fresh kernel, top to bottom, without reusing variables from an older session.
2. Confirm that every artifact for a section was produced by the same execution and records the expected dataset hash.
3. Replace the legacy `reports/section_NN` contents with the matching fresh export as one atomic set.
4. Reconcile the written report against the new metrics, confusion counts, thresholds, chosen epochs and limitations.
5. Shorten the final report to the formal 3,000-word limit unless an authoritative amendment is obtained.

## Verification status

Static verification passed after remediation: all four notebook files parse as JSON, all 60 ordinary Python cells compile, literal dictionaries contain no duplicate keys, execution counts are cleared, stored outputs are empty, and `git diff --check` reports no patch-format errors.

This static verification does not replace execution. Full training was not run during remediation because it requires the datasets, installed scientific packages and substantial compute. Runtime correctness and final numerical results must therefore be confirmed by the clean reruns described above.
