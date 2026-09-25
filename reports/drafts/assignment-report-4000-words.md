# Machine Learning in Cyber: Implementation and Critical Evaluation

**Expanded draft - updated 25 September 2026**

Main-text word count: **4,000**, excluding headings, citation markers, this editorial note, references and the evidence appendix. The main text is delimited by comments; the count uses whitespace-separated words after removing heading lines and citation markers (hyphenated expressions count as one word).

**Submission warning:** the extracted assignment brief specifies a maximum of **3,000 words**. This longer draft follows the requested length, not the formal submission limit. Shorten it or obtain an authoritative amendment before submission.

**Evidence policy:** Section 1 methods and results come from the fresh export under `section_01_results`, produced by the updated notebook on 25 September 2026. Sections 2-4 still use retained JSON evidence under `reports/section_02` through `reports/section_04`; those notebooks have since been revised but not rerun, so their saved results remain legacy evidence. The [alignment review](../../docs/assignment/LECTURE_NOTEBOOK_ALIGNMENT.md) records the remaining provenance limitations. Results from different executions are not combined within a model comparison.

<!-- report-body:start -->

## 1. Purpose and relationship to the lectures

This project investigates four cybersecurity applications: email classification, network intrusion classification, traffic anomaly detection, and ransomware identification from API calls. Each experiment compares a classical machine learning method with a neural network. The objective is not simply to maximize accuracy, but to explain how data quality, representation, class imbalance and evaluation design influence the usefulness of predictions. This reflects the lecturer's repeated emphasis on a justified workflow, understandable implementation and honest interpretation rather than impressive scores alone. [L01] [L06] [L11]

The implementation consists of four self-contained Jupyter notebooks. Downloading, profiling, preprocessing, training, plotting and exporting are placed inside each notebook, making the sequence of operations visible. This follows the practical Python and Colab workflow discussed in Lecture 2. Comments now explain the purpose of important operations, including label mappings, train-only fitting, weighted losses and checkpoint selection. They describe existing behaviour rather than asserting that recommended improvements have already been implemented. [L02]

All ten available lecture notes were reviewed: Lectures 1-7 and 9-11. Lecture 8 has no usable notes, so alignment with its content cannot be verified. The updated email experiment includes normalization, input exploration, validation-loss monitoring and an export. The other three notebooks have source-level remediations but still require fresh executions before their revised methods can replace the retained results. The review therefore distinguishes implemented techniques, observed evidence and proposed extensions instead of presenting every lecture example as compulsory. [A]

Section 1 is now a reconciled experiment: its run summary, metrics, learning history, profiles and figures were exported from the code. Existing Section 1 files under `reports/section_01` are superseded by that export. For Sections 2-4, this report continues to identify the retained results explicitly and does not claim that current sources reproduce them. [E1S] [E1H] [A]

## 2. Experimental design and evaluation principles

Each model pair uses shared partitions within its task. Email and API sequences use approximately 70% training, 15% validation and 15% testing after deduplication. The email pipeline removes duplicate normalized representations before splitting, while intrusion detection retains the official NSL-KDD test partition and takes validation records from its training partition. Anomaly detection uses a separate normal-only training pool. Fixed seeds make sampling and initialization more repeatable, although software versions and execution state can still influence results. [N1] [N2] [N3] [N4]

The distinction between training, validation and testing is central. Fitted vocabularies, document frequencies, scaling parameters and supervised feature selection use training data. In the fresh email run, validation loss selects the LSTM checkpoint and both classifiers select thresholds under a one-percent validation false-positive-rate budget. Test labels assess the locked systems. The retained Section 3 results came from an older run whose missingness filter examined the complete dataset; the revised notebook moves that decision inside the training boundary, but new Section 3 results are still required. [L06] [E1H] [A]

Accuracy counts all correct predictions, which can conceal failure on rare attacks. Precision measures how many positive alerts are correct; recall measures how many actual positives are detected. Their harmonic mean, F1, penalizes imbalance between these objectives. False-positive rate instead divides false alerts by all benign examples, making it particularly relevant to operational workload. A detector with high recall but excessive false positives can be unusable despite apparently identifying most attacks. [L02] [L06]

ROC curves evaluate ranking through true-positive and false-positive rates across thresholds. Precision-recall curves focus on positive-class retrieval, while average precision summarizes that ranking. Neither curve proves probability calibration or establishes an appropriate deployment threshold. Multiclass intrusion detection additionally uses macro averages, giving each class equal importance, and weighted F1, reflecting class support. These measures are complementary rather than interchangeable; results from different tasks should not be ranked as one league table. [L06]

Lecture 7 recommends comparing training and validation learning behaviour. All neural models use dropout, gradient-based optimization and validation-selected checkpoints. The updated email LSTM records training loss, validation loss and validation F1 for every epoch, then restores the lowest-validation-loss checkpoint after completing its fixed six-epoch budget. This is checkpoint selection, not early stopping. The other supervised neural results still require reconciled histories from fresh runs, so broader convergence comparisons remain premature. [L07] [E1H]

The lectures also distinguish statistical description from assumptions imposed by a model. Section 1 now plots class balance and normalized message length and records parsing and overlap diagnostics. Histograms, quantiles, skewness and correlations can likewise reveal long-tailed or redundant flow measurements, but they do not establish a causal attack mechanism. Neither standard scaling nor a high normality-test p-value proves a Gaussian distribution. Lecture 5's regression metrics are also not substitutes for classification metrics: reconstruction MSE is a training objective for the autoencoder, not a direct measure of threat-detection success. [L03] [L05] [E1P]

## 3. Email security: Logistic Regression and LSTM

Section 1 uses the Apache SpamAssassin easy-ham and spam archives. The fresh export identifies the two archive hashes and a hash of the extracted-file manifest. From 3,000 source messages, normalization-aware deduplication retains 2,947 records: 2,471 ham and 476 spam. The shared stratified split contains 2,062 training, 442 validation and 443 test messages; the test set contains 371 ham and 72 spam. These labels identify spam, not phishing specifically, so the experiment demonstrates text classification for email security without establishing phishing-detection performance. [D1] [E1S] [E1P]

The parser excludes attachments, combines the subject with one preferred body and selects plain text before falling back to HTML. A standard-library HTML parser extracts visible text rather than removing tags with a regular expression. The quality profile records 211 HTML fallbacks, three unknown-character-set fallbacks, one message without a text body and 3,701 decoding replacement characters. These counts expose malformed input, although replacement characters still represent lost information. [N1] [E1P]

Normalization lowercases text, replaces URLs and email addresses with `urltoken` and `emailtoken`, extracts Unicode word tokens and applies English Snowball stemming. The TF-IDF branch removes a stemmed list of 297 English stopwords while protecting `no`, `nor` and `not`; the LSTM keeps stopwords because order and negation context may be informative. No empty normalized messages or conflicting labels were found. Deduplicating normalized text before splitting leaves zero exact normalized-representation overlaps between every pair of partitions. This meets the brief's stemming and stopword requirements while reducing representation leakage, but cannot rule out similar campaigns. [L11] [B] [E1P] [E1O]

The classical pipeline learns training-only TF-IDF features using unigrams and bigrams. It caps the vocabulary at 20,000 features, excludes terms present in fewer than two documents or more than 98% of documents, and uses sublinear term frequency. Balanced Logistic Regression then learns a linear boundary with at most 1,000 iterations. Validation scores select threshold 0.5463 by maximizing F1 among thresholds whose validation false-positive rate is no greater than one percent. The threshold is locked before test evaluation. [N1] [E1C]

The neural pipeline builds a training-only vocabulary with padding and unknown tokens. Each message is truncated or padded to 300 tokens. Learned 128-dimensional embeddings feed a 128-unit LSTM, dropout of 0.3 and one output logit. Weighted binary cross-entropy increases the influence of the smaller spam class, and AdamW trains on the CPU for six epochs. Validation loss reaches its minimum, 0.2392, at epoch three, so that checkpoint is restored. A separate validation search selects threshold 0.9772 under the same one-percent false-alert budget. [L07] [L10] [N1] [E1D] [E1H]

The representations capture different evidence. TF-IDF emphasizes discriminative words and adjacent pairs but cannot model full message order. The LSTM learns its embeddings jointly and can represent ordered context, although its limited corpus may not justify the additional capacity. Neither output should be treated as a calibrated deployment probability: the thresholds are validation-selected operating points. The high LSTM threshold reflects the false-alert constraint and changes the precision-recall trade-off substantially compared with a fixed 0.5 cutoff. [L06] [L10]

The freshly exported test results favour Logistic Regression. It achieves accuracy 0.9842, precision 0.9710, recall 0.9306 and F1 0.9504. Its confusion matrix contains 369 correctly accepted ham messages, two false alerts, five missed spam messages and 67 detected spam messages. The LSTM achieves accuracy 0.9413, precision 1.0000, recall 0.6389 and F1 0.7797. It produces no false alerts but misses 26 of 72 spam messages, detecting 46. Under the shared validation false-alert policy, the linear model therefore offers the stronger balance between protecting legitimate email and detecting spam. [E1C] [E1D]

Ranking metrics tell a similar but less threshold-dependent story. Logistic Regression reaches ROC AUC 0.9996 and average precision 0.9977, compared with 0.9728 and 0.9444 for the LSTM. The LSTM training loss continues to fall after epoch three while validation loss rises before partially recovering, supporting restoration of the earlier checkpoint and suggesting overfitting rather than continued generalization. This interpretation concerns one seeded run, not a distribution across repeated experiments. [E1C] [E1D] [E1H]

The result remains limited by the historical corpus, spam/ham target and possible near-duplicate campaigns. Replacing URL and address identities with markers supports generalization but removes potentially useful structure. A focused error analysis should inspect the five Logistic Regression false negatives, two false positives and 26 LSTM false negatives for truncation, encoding damage, shared templates or indicators removed during normalization. Such analysis should use redacted examples, avoid tuning against the test set, and never reinterpret spam labels as verified phishing. [D1] [E1C] [E1D]

## 4. Network intrusion: Random Forest and CNN

Section 2 uses NSL-KDD, an assignment-suggested benchmark with 41 connection features. Three fields describe protocol, service and connection flag; the remaining predictors are numerical. Attack names are mapped into normal, denial of service, probe, remote-to-local and user-to-root categories. The attack label and difficulty metadata are excluded from model inputs. The official training file supplies 107,077 fitting records and 18,896 validation records, while all 22,544 official test records remain reserved for evaluation. [D2] [E2S]

Unlike the raw email and flow files, this benchmark is relatively prepared. The saved profile reports no missing values and no exact full-record duplicates within each partition. The current notebook therefore performs no imputation. This should be explained honestly instead of inventing missing-value treatment to appear comprehensive. Lecture 3 still motivates additional exploration of distributions, ranges, correlations and rare categories. Existing quality counts do not establish that predictor vectors are unique across partitions, because the duplicate check includes metadata and is performed separately. [L03] [D2P]

A training-fitted column transformer standardizes numerical features and one-hot encodes categorical fields, ignoring previously unseen categories during transformation. Zero-variance features are removed, followed by ANOVA selection of 64 features. This is feature selection, not PCA: original transformed variables are retained rather than combined into components. Standardization supports neural optimization but does not make skewed data Gaussian. Both models receive the same selected matrix, improving comparability while potentially constraining a forest that could exploit interactions outside the selected subset. [L04] [N2]

The Random Forest contains 180 trees, maximum depth 28 and square-root feature sampling. Balanced subsample weights address class imbalance within bootstrap samples. The CNN instead applies two one-dimensional convolutional layers with 32 and 64 channels, batch normalization and ReLU activations. Adaptive maximum pooling, dropout of 0.3 and a five-output layer complete the model. Weighted cross-entropy uses softened inverse-frequency class weights; Adam trains for eight epochs, selecting the checkpoint with highest validation macro F1. [N2]

The convolutional interpretation requires qualification. Its input axis is an ordered list of selected tabular features, not consecutive packets or a natural image grid. Neighbouring columns do not necessarily describe related measurements. Consequently, the CNN provides a neural comparison but should not be described as learning temporal traffic sequences. A comparison with a model designed for unordered tabular predictors could test whether the convolutional assumption helps, but that experiment has not been implemented. [L09]

Saved test accuracy improves from 0.7377 for Random Forest to 0.7644 for CNN. Macro precision changes from 0.7850 to 0.8310, macro recall from 0.4738 to 0.5784, and macro F1 from 0.4844 to 0.6036. Weighted F1 is 0.6955 and 0.7469 respectively. Macro average precision increases more modestly, from 0.6813 to 0.6960. Together, these figures favour the CNN in this recorded comparison, without proving a statistically reliable architecture advantage across alternative splits or seeds. [E2C] [E2D]

Per-class results expose the remaining weaknesses. The forest detects only 108 of 2,754 remote-to-local examples and four of 200 user-to-root examples. The CNN improves these counts to 779 and eighteen, corresponding to recalls of 0.2829 and 0.0900. Training contains only 846 remote-to-local and 44 user-to-root records. Class weighting cannot create missing behavioural diversity. High precision for a rare class can coexist with very low recall when the classifier makes few positive predictions. [D2P] [E2C] [E2D]

Intrusion errors also have two meanings. Confusing one attack family with another can still trigger an investigation, whereas assigning an attack to normal may suppress an alert completely. The five-class evaluation preserves that distinction in its confusion matrix, but it does not separately report a collapsed normal-versus-attack operating curve. For example, the CNN assigns 1,854 remote-to-local records to normal, demonstrating substantial missed malicious activity beyond its family-classification score. Any later binary analysis should be clearly labelled as a different evaluation target. [E2D]

The CNN also reduces normal-class recall from 0.9745 to 0.9229 and denial-of-service recall from 0.7591 to 0.7167, while improving probe recall from 0.5762 to 0.8794. Thus, aggregate improvement includes trade-offs rather than universal gains. The benchmark's age, scarce classes and differences between official training and test attacks limit generalization claims. Appropriate next steps include training-only resampling studies, better feature analysis and external evaluation, without repeatedly selecting changes against this already inspected test set. [E2C] [E2D] [D2]

## 5. Traffic anomalies: Isolation Forest and Autoencoder

Section 3 provides the clearest example of genuine data cleaning. It uses the unmodified CSE-CIC-IDS2018 flow-feature CSV for 1 March 2018, containing benign and infiltration traffic. The raw file has 331,125 records and 80 columns, including 1,834 missing values, 4,000 infinite numeric values, 25 embedded headers and 97 duplicate rows. These are extracted traffic features rather than raw packet captures, but the file is demonstrably not model-ready. [D3]

Cleaning normalizes column names, removes embedded headers and then deletes 73 remaining duplicate rows. Twenty-four repeated headers were themselves duplicates, explaining why 97 initial duplicates do not become 97 later removals. Infinities become missing values, leaving 5,834 missing entries before imputation. The cleaned dataset contains 331,027 records: 237,987 benign and 93,040 infiltration. Statistical outliers are deliberately retained because unusual behaviour may be the phenomenon being detected, rather than a disposable error. [L03] [L09] [D3]

The notebook selects flow measurements and derives hour and weekday from timestamps. It samples 120,000 benign and 18,000 infiltration records for practical execution. Training uses 72,000 benign flows; validation and testing each contain 24,000 benign and 9,000 infiltration flows. Labels therefore identify the normal training pool and construct evaluation partitions, even though neither model receives attack labels in its fitting objective. Labelled validation also supports threshold selection, so this is not a wholly label-free detection study. [N3]

Median imputation with missingness indicators, zero-variance filtering and standardization are fitted on benign training data. The resulting matrix has 67 features. This preserves missingness information while avoiding evaluation-driven medians or scaling statistics. However, the earlier high-missingness feature filter sees the complete dataset. Numerical treatment of protocol codes and destination ports also deserves stronger semantic justification. A single randomly partitioned day can share environmental patterns across all splits; it does not demonstrate future-day or unseen-family generalization. [L04] [N3]

Isolation Forest learns 200 randomized isolation trees using up to 10,000 observations per tree. Shorter isolation paths can indicate unusual points. The implementation negates its normality scores so that larger values consistently mean more anomalous behaviour. Although contamination is set to automatic, final predictions do not use that default boundary. Instead, the selected threshold maximizes F1 on labelled validation scores before being applied to the test set. [L09] [N3]

The autoencoder learns to reconstruct standardized benign inputs through layers of 64, 32 and twelve units before expanding symmetrically. The twelve-unit bottleneck restricts representation capacity. Hidden ReLU activations and dropout support learning and regularization; a linear output accommodates signed standardized values. Adam minimizes mean squared reconstruction error for twelve epochs. The checkpoint with lowest benign-validation reconstruction loss is restored. Each flow's anomaly score is its mean squared feature reconstruction error, and a separate validation F1 threshold determines alerts. [L07] [N3]

Saved Isolation Forest results show true-positive rate 0.9558, false-positive rate 0.9156, precision 0.2813 and F1 0.4347. The autoencoder reaches true-positive rate 0.9807, false-positive rate 0.8754, precision 0.2958 and F1 0.4545. The apparent detection sensitivity is misleading without the false-alert counts: Isolation Forest flags 21,975 benign flows, while the autoencoder flags 21,009. Both would impose an overwhelming investigative burden. These results should be reported as an operational failure, not disguised by emphasizing recall alone. [E3C] [E3D]

Ranking metrics reinforce this conclusion. Isolation Forest has ROC AUC 0.4606 and average precision 0.2474; the autoencoder has 0.5176 and 0.2754. Anomalies constitute 27.27% of the test set, providing a prevalence reference for precision-recall interpretation. An always-alert rule would have recall one, precision 0.2727 and F1 approximately 0.4286. The measured F1 improvements over that trivial rule are small. Strong reconstruction of benign traffic is therefore not equivalent to reliable separation of labelled attacks. [E3C] [E3D]

The autoencoder's validation criterion and detection criterion are deliberately different. Benign reconstruction loss chooses weights that represent normal traffic, while labelled validation F1 chooses an alert threshold. A falling reconstruction curve therefore cannot by itself demonstrate improved attack ranking. Furthermore, dropout is active during fitting but disabled during scoring, so training and validation losses are computed under different conditions. Reusing one validation partition for checkpoint and threshold selection is permissible for this demonstration, but additional development would benefit from more independent calibration evidence. [N3]

A plausible explanation is substantial overlap between infiltration behaviour and benign flow summaries, potentially compounded by feature choices or dataset artefacts. The current evidence does not isolate those causes. Selecting thresholds to maximize F1 also imposes no explicit limit on false alerts. Future work should examine validation thresholds constrained by an operational false-positive budget, score distributions, meaningful protocol encodings and cross-day evaluation. Improving the reported score by tuning against the final test labels would undermine the evaluation rather than solve the detection problem. [L06] [L09]

## 6. Ransomware: API n-grams and sequence learning

Section 4 uses ordered Windows API-call traces from the locally documented MalwareAPI dataset. Only Goodware and ransomware labels are retained from 41,476 source rows, yielding 1,374 selected records. Removing missing entries and duplicate API sequences leaves 630 examples: 591 goodware and 39 ransomware. Every selected trace contains 100 calls. This makes the dataset relevant to behaviour-based classification, but the available observations are call sequences rather than complete system histories, file contents or direct evidence of prevented encryption. [D4] [N4]

The small unique ransomware class is the main statistical constraint. Training contains 440 examples, including 27 ransomware traces; validation and testing each contain 95 examples, including six ransomware traces. One changed positive prediction therefore moves test recall by approximately 16.7 percentage points. Deduplication reduces direct sequence leakage, but it does not guarantee independence between related malware families or variants. Random stratification also cannot demonstrate detection of future ransomware families. [D4] [E4S]

A read-only review identified an additional quality issue: two unique sequences carry both Goodware and ransomware labels, involving eight source rows. The current keep-first deduplication retains one under each class, letting source row order resolve the ambiguity. This is not a justified labelling policy. The experiment and saved results were preserved, but a revised study should investigate or exclude ambiguous traces before constructing partitions. Its results would then need regeneration rather than silent reuse of the present metrics. [A]

The classical representation uses case-preserving TF-IDF over API unigrams and bigrams, capped at 5,000 features with sublinear term frequency. These features capture call presence and short transitions without representing the entire call order. A balanced linear SVM learns a separating margin. Its decision scores are transformed with a sigmoid for the shared evaluator. That transformation preserves ranking and makes the 0.5 threshold equivalent to zero margin, but it does not create calibrated ransomware probabilities. [N4]

The LSTM learns a training-only API vocabulary, 64-dimensional embeddings and a 64-unit recurrent representation. Dropout of 0.3 precedes the binary output, and the final real timestep summarizes each sequence. Weighted binary cross-entropy addresses imbalance using the training goodware-to-ransomware ratio; AdamW updates the parameters. Validation F1 chooses the checkpoint. The saved comparison records fifteen epochs, whereas the current notebook and its stored output use twelve, so the two runs must not be treated as identical evidence. [L10] [E4D]

In the saved results, SVM precision is 1.0000, recall 0.3333 and F1 0.5000. The LSTM has precision 0.2500, recall 0.3333 and F1 0.2857. Both detect only two of six ransomware traces and miss four. The SVM produces no false positives among 89 goodware traces; the LSTM produces six. Perfect observed SVM precision therefore means only that its two positive predictions were correct, not that the detector is generally reliable. [E4C] [E4D]

SVM accuracy is 0.9579 compared with LSTM accuracy 0.8947. An always-goodware classifier would already achieve 89/95, or approximately 0.9368, while detecting no ransomware. This demonstrates why accuracy alone is especially misleading here. ROC AUC favours SVM, 0.9270 versus 0.8296, and average precision is 0.6110 versus 0.4546. These ranking differences favour the classical model in the saved sample, but six positive test cases cannot support confident population-level superiority claims. [E4C] [E4D]

The observed zero SVM false positives is also uncertain: it describes only 89 tested goodware traces, not every legitimate application. A larger, independent goodware sample is needed to assess rare false alerts. Likewise, neural checkpoint selection against six validation positives can change sharply after one prediction changes. Repeated, appropriately grouped evaluation and uncertainty intervals would be more informative than adding decimal places to F1. No such repeated evaluation or interval estimation was performed, and neither is implied by the rounded metrics. [E4C]

The implementation detects patterns in completed traces; it does not implement prevention. No measurement establishes how early malicious activity becomes detectable or whether intervention precedes file damage. Process isolation, protected backups, least privilege and analyst review are sensible proposed layers, not evaluated outcomes of these notebooks. A credible prevention study would require timestamped behaviour, response-cost analysis and controlled intervention evidence alongside broader, consistently labelled ransomware examples. [N4]

## 7. Cross-experiment assessment and conclusion

The results do not support the assumption that a more complex neural model necessarily performs better. Logistic Regression leads the email comparison, the CNN improves intrusion macro F1, the autoencoder only slightly improves poor anomaly discrimination, and SVM leads the small ransomware experiment. These outcomes reinforce the lecturer's emphasis on representation, suitable data and evaluation rather than architecture alone. Because preprocessing branches and training procedures differ, the comparisons assess complete pipelines rather than isolated causal effects of model families. [L04] [L07]

The recorded experiments are fixed comparisons rather than comprehensive model-selection studies. Classical hyperparameters are specified directly, while neural checkpoints are selected using validation performance. There is no documented search over tree depth, vocabulary size, recurrent capacity or learning rate, and no ablation isolates the effect of class weights. Therefore, the findings describe these particular configurations, not the best achievable performance of each algorithm. A fair extension would define comparable tuning budgets in advance, fit every transformation inside each training partition, select configurations using validation evidence, and reserve an independent holdout for the final comparison. Both unsuccessful and successful configurations should remain visible in the final experiment record.

Reproducibility is now substantially stronger for Section 1. Its export records archive and corpus-manifest hashes, the seed, Python and package versions, CPU execution, split membership, preprocessing profiles, numerical learning history, selected epoch, metrics and figures. Sections 2-4 still require equivalent fresh exports before their revised code can be connected to final results. Runtime, memory consumption and inference latency were not measured, so efficiency claims would remain speculative. [E1S] [E1H] [A]

The immediate priority is to rerun Sections 2-4 and reconcile each notebook with one final set of splits, parameters, histories and artifacts. Those runs should confirm the revised ransomware conflict policy, training-only anomaly feature filtering, richer distribution analysis and supervised validation-loss monitoring. Error examples and uncertainty estimates would make every evaluation more informative. Any development prompted by inspected results should be followed by genuinely independent evaluation rather than repeated optimization on the same test examples. [A]

Finally, comparison should include the information available to each pipeline, not only its final architecture. The email branches use different stopword policies, and the anomaly systems respond to different notions of unusualness. Recording these choices makes a negative result interpretable. A future experimental register should distinguish a planned change, its validation rationale, the frozen configuration and the independent evaluation outcome. This would connect the lecturer's methodological advice to an auditable development process without turning the notebooks into an unnecessarily general software framework.

Overall, the notebooks demonstrate the central taught workflow and expose meaningful security trade-offs, but they remain educational experiments. Strong spam discrimination does not establish phishing detection; rare intrusion classes remain poorly covered; anomaly alerts are excessive; and ransomware evidence is limited. Reporting these weaknesses explicitly is more defensible than claiming deployment readiness or treating every high score as successful cyber defence.

<!-- report-body:end -->

## Evidence appendix — excluded from main-text word count

Section 1 values are rounded from the fresh `section_01_results` export; Sections 2-4 use the retained JSON artifacts linked below. The report-writing update did not rerun any model. AP means average precision, not trapezoidal PR area. Binary metrics use spam, anomaly or ransomware as the positive class.

| Task/model | Precision | Recall | F1 | ROC AUC | AP |
|---|---:|---:|---:|---:|---:|
| Email: Logistic Regression | 0.9710 | 0.9306 | 0.9504 | 0.9996 | 0.9977 |
| Email: LSTM | 1.0000 | 0.6389 | 0.7797 | 0.9728 | 0.9444 |
| Anomalies: Isolation Forest | 0.2813 | 0.9558 | 0.4347 | 0.4606 | 0.2474 |
| Anomalies: Autoencoder | 0.2958 | 0.9807 | 0.4545 | 0.5176 | 0.2754 |
| Ransomware: SVM | 1.0000 | 0.3333 | 0.5000 | 0.9270 | 0.6110 |
| Ransomware: LSTM, saved 15-epoch run | 0.2500 | 0.3333 | 0.2857 | 0.8296 | 0.4546 |

| Intrusion model | Accuracy | Macro precision | Macro recall | Macro F1 | Weighted F1 | Macro AP |
|---|---:|---:|---:|---:|---:|---:|
| Random Forest | 0.7377 | 0.7850 | 0.4738 | 0.4844 | 0.6955 | 0.6813 |
| CNN | 0.7644 | 0.8310 | 0.5784 | 0.6036 | 0.7469 | 0.6960 |

For anomalies, false-positive rate is **0.915625** for Isolation Forest and **0.875375** for Autoencoder. High recall in the table must be interpreted alongside those false-alert rates.

### Existing figures

The Section 1 links identify figures from the fresh export. Sections 2-4 link retained figures whose provenance must be reconciled by their pending reruns; no cross-run learning-curve conclusions are made.

- Email: [data exploration](../../section_01_results/reports/figures/data-exploration.png), [classical evaluation](../../section_01_results/reports/figures/classic-evaluation.png), [LSTM evaluation](../../section_01_results/reports/figures/lstm-evaluation.png), [LSTM training history](../../section_01_results/reports/figures/lstm-training-history.png).
- Intrusion: [Random Forest evaluation](../section_02/figures/random-forest-evaluation.png), [CNN evaluation](../section_02/figures/cnn-evaluation.png).
- Anomalies: [Isolation Forest evaluation](../section_03/figures/isolation-forest-evaluation.png), [Autoencoder evaluation](../section_03/figures/autoencoder-evaluation.png).
- Ransomware: [SVM evaluation](../section_04/figures/svm-evaluation.png), [LSTM evaluation](../section_04/figures/lstm-evaluation.png).

## References and implementation evidence

Lecture citations refer to the repository's recording-derived notes, not independently verified transcripts. Implementation and numerical claims refer to the linked local files. Dataset provenance and original publisher links are available in the four dataset records.

### Teaching and assignment

- [L01 — Introduction to machine learning][L01]
- [L02 — Python, Colab and decision trees][L02]
- [L03 — Preprocessing and probability distributions][L03]
- [L04 — Feature engineering][L04]
- [L05 — Regression][L05]
- [L06 — Classification and evaluation][L06]
- [L07 — Artificial neural networks][L07]
- [L09 — Anomaly detection and CNNs][L09]
- [L10 — Embeddings, RNNs and BERT][L10]
- [L11 — NLP and assignment explanation][L11]
- [B — Extracted assignment brief][B]
- [A — Detailed lecture–notebook alignment audit][A]

### Notebooks, datasets and results

- Section 1: [notebook][N1], [dataset record][D1], [fresh run summary][E1S], [text profile][E1P], [overlap checks][E1O], [training history][E1H], [Logistic Regression metrics][E1C], [LSTM metrics][E1D].
- Section 2: [notebook][N2], [dataset record][D2], [saved data profile][D2P], [run summary][E2S], [Random Forest metrics][E2C], [CNN metrics][E2D].
- Section 3: [notebook][N3], [dataset record][D3R], [saved cleaning profile][D3], [run summary][E3S], [Isolation Forest metrics][E3C], [Autoencoder metrics][E3D].
- Section 4: [notebook][N4], [dataset record][D4], [run summary][E4S], [SVM metrics][E4C], [LSTM metrics][E4D].

[L01]: ../../docs/lectures/lecture-01-2026-07-12-introduction-to-machine-learning.md
[L02]: ../../docs/lectures/lecture-02-2026-07-19-python-colab-and-decision-trees.md
[L03]: ../../docs/lectures/lecture-03-2026-07-26-data-preprocessing-and-probability-distributions.md
[L04]: ../../docs/lectures/lecture-04-2026-08-02-feature-engineering.md
[L05]: ../../docs/lectures/lecture-05-2026-08-09-regression.md
[L06]: ../../docs/lectures/lecture-06-2026-08-16-classification-and-evaluation.md
[L07]: ../../docs/lectures/lecture-07-2026-08-23-artificial-neural-networks.md
[L09]: ../../docs/lectures/lecture-09-2026-09-06-anomaly-detection-and-cnns.md
[L10]: ../../docs/lectures/lecture-10-2026-09-13-word-embeddings-rnns-and-bert.md
[L11]: ../../docs/lectures/lecture-11-2026-09-20-natural-language-processing.md
[B]: ../../docs/assignment/ASSIGNMENT_INFORMATION.md
[A]: ../../docs/assignment/LECTURE_NOTEBOOK_ALIGNMENT.md
[N1]: ../../notebooks/01_phishing/section-01-email-security.ipynb
[N2]: ../../notebooks/02_intrusion_detection/section-02-intrusion-detection.ipynb
[N3]: ../../notebooks/03_anomaly_detection/section-03-anomaly-detection.ipynb
[N4]: ../../notebooks/04_ransomware/section-04-ransomware-detection.ipynb
[D1]: ../../docs/research/spamassassin-public-corpus.md
[D2]: ../../docs/research/nsl-kdd-dataset.md
[D2P]: ../../data/processed/section_02/data-profile.json
[D3R]: ../../docs/research/cic-ids2018-anomaly-dataset.md
[D3]: ../../data/processed/section_03/data-quality-report.json
[D4]: ../../docs/research/ransomware-api-sequence-dataset.md
[E1S]: ../../section_01_results/reports/run-summary.json
[E1C]: ../../section_01_results/reports/metrics/classic.json
[E1D]: ../../section_01_results/reports/metrics/lstm.json
[E1P]: ../../section_01_results/processed/text-profile.json
[E1O]: ../../section_01_results/processed/representation-overlap.json
[E1H]: ../../section_01_results/reports/metrics/lstm-training-history.csv
[E2S]: ../section_02/run-summary.json
[E2C]: ../section_02/metrics/random-forest.json
[E2D]: ../section_02/metrics/cnn.json
[E3S]: ../section_03/run-summary.json
[E3C]: ../section_03/metrics/isolation-forest.json
[E3D]: ../section_03/metrics/autoencoder.json
[E4S]: ../section_04/run-summary.json
[E4C]: ../section_04/metrics/svm.json
[E4D]: ../section_04/metrics/lstm.json
