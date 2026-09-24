# Lecture–notebook alignment review

Reviewed 24 September 2026. Scope: all ten available lecture-note files, all four current notebooks (including stored text outputs), the extracted assignment brief, dataset records, saved profiles, metric JSON files, and older report drafts. Lecture 8 has no usable notes in the [lecture index](../lectures/README.md), so its content cannot be assessed.

## Overall finding

**The experiments substantially follow the taught workflow, but they are not fully aligned or submission-ready.** Each section compares an appropriate classical model with a neural model, separates fitting from evaluation, and reports security-relevant metrics. Important gaps concern data exploration, Section 1 text processing, result provenance, Section 3 preprocessing boundaries, and Section 4 label quality/sample size.

This review added explanatory comments to every code cell. It did **not** retrain models, change algorithms, resolve labels, alter thresholds, or overwrite existing notebook outputs/results. The [expanded report](../../reports/drafts/assignment-report-4000-words.md) explains the implementation as it actually exists, rather than claiming the gaps have been fixed.

The formal [brief](ASSIGNMENT_INFORMATION.md) limits the report to **3,000 words**. Lecture 11 mentions a possible longer explanation, but that is not an authoritative amendment. The requested 4,000-word report is therefore an expanded draft to shorten or obtain approval for before submission.

## Coverage of every available lecture

| Lecture | Relevant teaching | Notebook alignment and remaining gap |
|---|---|---|
| [1: Introduction](../lectures/lecture-01-2026-07-12-introduction-to-machine-learning.md) | Learning from data, supervised/unsupervised learning, generalization | Sections 1, 2 and 4 are supervised; Section 3 fits normal-only models. Holdouts exist, but none establishes production or zero-day performance. |
| [2: Python, Colab and trees](../lectures/lecture-02-2026-07-19-python-colab-and-decision-trees.md) | Notebook workflow, features versus targets, trees, confusion matrices, misleading accuracy | All four are self-contained notebooks with confusion matrices. Section 2 supplies the tree ensemble. Sections 1–3 begin with an unconditional Colab-only working directory. |
| [3: Preprocessing and distributions](../lectures/lecture-03-2026-07-26-data-preprocessing-and-probability-distributions.md) | Inspect types, missingness, duplicates, distributions, skewness and outliers; split appropriately | Counts/quality checks and splits are present. Section 3 has substantial real cleaning. Systematic feature summaries, skewness, correlation and distribution plots remain limited. Anomaly-score histograms are not input-feature EDA. |
| [4: Feature engineering](../lectures/lecture-04-2026-08-02-feature-engineering.md) | Encoding, scaling, feature selection versus extraction, meaningful behaviour features | TF-IDF, embeddings, one-hot encoding, ANOVA selection, missingness indicators and temporal features are used. Section 3 treats protocol codes/ports numerically; temporal and interaction feature choices need stronger justification. |
| [5: Regression](../lectures/lecture-05-2026-08-09-regression.md) | Reproducible splitting, exploration even for clean data, model complexity | Seeds and partitions align. Regression prediction metrics such as R² are not required for these classification tasks; autoencoder reconstruction MSE has a different role. Exploration is still incomplete. |
| [6: Classification and evaluation](../lectures/lecture-06-2026-08-16-classification-and-evaluation.md) | Imbalance, precision/recall/F1, validation, train-only preprocessing, threshold trade-offs | Class weighting, macro metrics and PR/ROC analysis align. Most learned transforms are train-only. Section 3's pre-split missingness filter is an exception; Sections 1/4 retain a fixed threshold without validation cost analysis. |
| [7: Neural networks](../lectures/lecture-07-2026-08-23-artificial-neural-networks.md) | Loss, backpropagation, activations, dropout, training versus validation curves | Neural implementations use these mechanisms. Only Section 3 tracks validation loss; Sections 1/2/4 track validation F1 instead. Best-checkpoint restoration is implemented, **not early stopping**. |
| [9: Anomalies and CNNs](../lectures/lecture-09-2026-09-06-anomaly-detection-and-cnns.md) | Normality versus maliciousness, isolation, thresholding, convolutions | Section 3 follows normal-only fitting and validation threshold selection. Its FPR is operationally unacceptable. Section 2 convolves over selected tabular columns with no natural spatial ordering. |
| [10: Embeddings, RNNs and BERT](../lectures/lecture-10-2026-09-13-word-embeddings-rnns-and-bert.md) | Token identity, learned embeddings, ordered context, recurrent memory, cost | Sections 1/4 use train-only vocabularies, learned embeddings and LSTMs. They do not use pretrained embeddings or BERT, which are alternatives, not mandatory additions. Runtime/memory comparisons were not measured. |
| [11: NLP and assignment guidance](../lectures/lecture-11-2026-09-20-natural-language-processing.md) | Domain-aware cleaning, stopwords, stemming/lemmatization, explain decisions and failures | Section 1 replaces URLs/addresses and removes punctuation; only TF-IDF removes stopwords. Neither branch stems/lemmatizes despite the brief. The new report explains limitations and links code/results; current evidence needs reconciliation. |

PCA, BERT, normality tests, a particular 80/20 split, and every candidate feature mentioned in lectures are **not all compulsory**. The important distinction is between an explicit brief requirement, a teaching recommendation, and an illustrative alternative. Fixed 70/15/15 splits are defensible; choosing seed 42 is a reproducibility convention, not a requirement.

## Section-by-section review

### 1. Email security

[Notebook](../../notebooks/01_phishing/section-01-email-security.ipynb): MIME parsing and malformed-charset fallback, attachment exclusion, text deduplication, stratified shared splits, train-only TF-IDF/vocabulary, weighted Logistic Regression and LSTM, confusion/ROC/PR plots.

- **Explicit gap:** `normalize` performs neither stemming nor lemmatization. English stopword removal is applied by TF-IDF only, not by the LSTM tokenizer. Documenting this is not the same as satisfying the brief.
- **Target limitation:** SpamAssassin labels mean spam/ham, not phishing/legitimate. Keep those meanings in every result claim.
- **Cleaning limitation:** `read_email` concatenates all non-attachment text parts, including plain/HTML alternatives. It does not prefer plain text as the older dataset note says. Regex HTML removal is not full HTML parsing.
- **Information loss:** normalization removes URL/address identity, punctuation and non-ASCII letters. Deduplication precedes normalization, so identical model representations or near-duplicate campaigns can still cross partitions.
- **Evidence gap:** inspect message lengths, charset failures, empty normalized text, conflicting labels and normalized-text overlap. Review the generic stopword list against security language. These checks are not currently complete.

### 2. Intrusion detection

[Notebook](../../notebooks/02_intrusion_detection/section-02-intrusion-detection.ipynb): official NSL-KDD test partition; training-only scaling/one-hot encoding/variance filtering/ANOVA; 64 selected features shared by Random Forest and CNN; five-class metrics and PR curves.

- **Clean data is acceptable if explained:** recorded profiles contain zero missing values and zero within-partition full-record duplicates. The current pipeline performs **no imputation**; the older draft's imputation claim is stale.
- **Profiling boundary:** duplicate checks include attack name and difficulty, and check within each split. They do not prove absence of duplicate predictor vectors or cross-partition overlap.
- **Model interpretation:** the CNN's axis contains feature columns, not successive packets. Discuss its locality assumption rather than describing it as temporal traffic modelling.
- **Evaluation:** rare-class recall is essential. The saved CNN detects only 18/200 U2R records despite outperforming the forest on macro F1. No probability calibration or hyperparameter-search study is implemented.

### 3. Anomaly detection

[Notebook](../../notebooks/03_anomaly_detection/section-03-anomaly-detection.ipynb): genuine dirty flow CSV; header/duplicate cleaning; infinity replacement; benign-only training; fitted median imputation with missingness indicators; standardization; Isolation Forest and Autoencoder; validation F1 thresholds; TPR/FPR/PR evaluation.

- **Preprocessing boundary:** `numeric.isna().mean() > 0.4` examines all records before splitting. This is a data-dependent feature-selection leakage risk even without labels. The saved quality report shows no features were removed, so do not claim a measured inflation of these results. Fit any such selection on training records in a revised experiment.
- **Label use:** labels select benign training records, construct the evaluation pools, identify benign validation records for checkpoint selection, and tune the threshold. They do not enter the model fitting loss. The notebook's existing statement that labels are used “only” for thresholds/evaluation is too broad.
- **Validity:** one randomly split day and labelled threshold selection do not establish unseen-family/zero-day detection. Exact full-row deduplication does not prove distinct feature representations across partitions.
- **Feature reasoning:** integer protocol/port values are scaled as continuous measurements. Consider semantic encoding; investigate constant or environment-specific timestamp features. Outliers are appropriately retained.
- **Result:** high TPR coexists with FPR above 87%; this is a negative operational result, not successful detection based on recall alone.

### 4. Ransomware

[Notebook](../../notebooks/04_ransomware/section-04-ransomware-detection.ipynb): ordered API traces; Goodware/ransomware filtering; exact sequence deduplication; shared splits; train-only API TF-IDF and LSTM vocabulary; balanced linear SVM and weighted LSTM.

- **Confirmed conflicting labels:** read-only profiling of the local raw CSV found **two unique API sequences with both labels**, involving eight rows: three Goodware and five ransomware. `drop_duplicates("APISEQ")` retains the first row, leaving one ambiguous sequence under each class. Excluding both ambiguous sequences would leave 590 goodware and 38 ransomware sequences (628 total), but this review did not change the data or reported experiment.
- **Statistical limitation:** the retained 630 sequences contain only 39 ransomware examples; validation/test each contain six. One prediction changes recall by 16.7 percentage points. Deduplication does not establish malware-family independence.
- **Scores:** `sigmoid(svm_scores)` is a monotonic display transformation, **not calibrated probability**. A threshold of 0.5 is equivalent to a zero SVM margin.
- **Detection versus prevention:** the notebook classifies completed API traces; it implements no live blocking, encryption rollback, or time-to-detection experiment. Prevention controls in its interpretation are proposals.

The conflicting-label finding can be reproduced without executing a notebook or fitting a model:

```python
import pandas as pd

raw = pd.read_csv("data/raw/ransomware-api/multiclass_malware_api_seq.csv")
selected = raw[raw["TYPE"].isin(["Goodware", "ransomware"])]
label_counts = selected.groupby("APISEQ")["TYPE"].nunique()
conflicting = selected[selected["APISEQ"].isin(label_counts[label_counts > 1].index)]
print(len(label_counts[label_counts > 1]))  # 2 distinct conflicting traces
print(conflicting["TYPE"].value_counts())  # ransomware: 5; Goodware: 3
```

## Results provenance: do not mix runs

The expanded report consistently uses `reports/section_NN/metrics/*.json` and associated saved summaries/profiles. These are retained artifact results, **not results newly generated by this review**. Notebook output cells contain different observations:

| Evidence | Saved JSON used in report | Stored notebook text output |
|---|---:|---:|
| Section 1 LSTM test F1 | 0.842767 | 0.936170 |
| Section 2 Random Forest macro F1 | 0.484418 | 0.481938 |
| Section 2 CNN macro F1 | 0.603553 | 0.570231 |
| Section 3 Isolation Forest threshold | 0.318722 | 0.317075 |
| Section 3 Autoencoder F1 | 0.454538 | 0.454679 |
| Section 4 LSTM epochs | 15 | 12, also the current source setting |
| Section 4 LSTM F1 | 0.285714 | 0.235294 |

Do not average these runs, attach an output-cell history to an unrelated saved metric, or claim current notebooks exactly reproduce the report. The cause of each difference has not been established. Seeds alone do not rule out changed code, environments, devices or execution state. Choose one final experiment version and regenerate/export its artifacts together before submission.

Older drafts are also inconsistent: Section 1 mentions early stopping; Section 2 describes imputation, early stopping and a one-epoch CNN; Section 3 lists different autoencoder results and says FPR exceeds 90% for both models. The current code runs fixed epoch budgets with checkpoint selection, and saved autoencoder FPR is 87.54%. Use the expanded report instead of copying those assertions.

## Prioritized follow-up, not implemented by this review

1. Reconcile notebook outputs, code parameters, profiles, figures and saved metrics under a single run identifier; retain package versions, dataset hashes, seeds, device and split membership. Section 4's 12/15 epoch mismatch is explicit.
2. Review the two conflicting API traces; choose and justify an exclusion/grouping or label-resolution policy, then rerun affected experiments. Do not silently assign labels by row order.
3. Address Section 1's explicit stemming/lemmatization requirement and explain any branch-specific stopword policy. Preserve the spam-versus-phishing distinction.
4. Fit Section 3's missingness feature filter on training data; add meaningful feature-distribution, skewness/correlation, representation-overlap and error-example analysis where appropriate.
5. Add validation-loss curves for the three supervised neural classifiers; save numerical histories and the chosen epoch. A validation-F1 curve does not replace a validation-loss curve.
6. For an operational study, choose validation thresholds against a justified false-alert budget and evaluate on an independent holdout. Do not optimize using the already inspected test metrics.
7. Fix/guard the unconditional `/content` changes in Sections 1–3 for local use. Export processed profiles/manifests as well as reports/models; package the fitted preprocessor with CNN/Autoencoder weights. Pin runtime versions if reproducibility is required.
8. Shorten the report to the formal limit or obtain an authoritative amendment. Keep unimplemented improvements clearly labelled as future work.

## Review verification

Static checks passed: all four notebook JSON files parse, all 46 code cells now have explanatory comments, and parsed Python syntax trees are unchanged after commenting (`%pip` is excluded from Python AST parsing). Stored outputs, execution counts and metadata are unchanged. The edits add 175 comment lines without changing executable statements or installation commands. Local document links and source references resolve, binary metric JSON values agree with their confusion counts, and the report contains exactly 4,000 main-text words under its stated counting rule.

No model training, notebook execution, package installation, new dataset download or result replacement was performed. These checks do not establish runtime correctness or reconcile the different saved runs. Comments explain the actual operations and identify known caveats; they do not repair those caveats.
