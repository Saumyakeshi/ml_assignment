# Section 3 Report Draft - Anomaly Detection for Cyber Threats

## Aim

This experiment investigated whether unsupervised models trained only on benign network flows could identify unusual behaviour representing previously unseen cyber threats. A classical Isolation Forest was compared with a denoising Autoencoder using the UNSW-NB15 dataset.

## Dataset and cleaning

UNSW-NB15 contains normal traffic and nine attack families: Analysis, Backdoor, DoS, Exploits, Fuzzers, Generic, Reconnaissance, Shellcode, and Worms. The published development and test CSV files originally contained 175,341 and 82,332 records respectively, with 45 columns each.

The pipeline removed exact duplicates using feature values rather than row identifiers. This removed 74,301 development records and 28,386 test records. A further 1,302 test records were removed because their feature representations also appeared in development data. The cleaned partitions contained 101,040 development records and 52,644 test records.

Numerical fields were median-imputed, transformed with signed `log1p`, and robustly scaled. Protocol, service, and connection state were one-hot encoded. Zero-variance transformed features were removed, leaving 58 model features. Every preprocessing operation was fitted using benign training records only.

## Experimental design

The training partition contained 15,000 benign records and no attacks. Validation contained 12,000 benign and 12,000 attack records. It was used to select model settings and choose the lowest-false-positive threshold that detected at least 60% of validation attacks. The cleaned official test partition contained 33,662 benign and 18,982 attack records and was used only for final evaluation.

This design preserves the unsupervised anomaly-detection objective: neither model receives attack examples during parameter fitting. Labels are used only for validation decisions and final evaluation.

## Models

Isolation Forest used 300 randomized trees. Six combinations of training sample size and feature proportion were compared using validation average precision, with ROC AUC as a secondary measure. The selected configuration used all 15,000 benign training records and all transformed features.

The denoising Autoencoder used a `128 -> 64 -> 16 -> 64 -> 128` hidden structure. It reconstructed clean benign records from slightly noisy inputs. Layer normalization, dropout, AdamW optimization, gradient clipping, Smooth L1 loss, and benign-validation monitoring were used for stable training. It trained for 30 epochs and retained the epoch with the lowest benign validation loss. Reconstruction error was normalized per feature before calculating each record's anomaly score.

## Results

| Model | Accuracy | TPR | FPR | Precision | F1 | Average precision | ROC AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| Isolation Forest | 0.729 | 0.615 | 0.206 | 0.627 | 0.621 | 0.672 | 0.763 |
| Denoising Autoencoder | 0.805 | 0.709 | 0.142 | 0.739 | 0.724 | 0.796 | 0.876 |

The Autoencoder performed best across every ranking and threshold-dependent measure. Its ROC AUC of 0.876 indicates that attack records generally receive higher anomaly scores than benign records. Average precision of 0.796 confirms useful performance under the observed class distribution. At the selected threshold, the model detected 13,465 of 18,982 attacks while producing 4,766 false alerts from 33,662 benign records.

The Isolation Forest also produced a useful ranking and now exceeds the required 50% attack recall. It detected 11,666 attacks and missed 7,316 at the selected threshold.

## Discussion and cybersecurity implications

The Autoencoder's stronger result suggests that learning a nonlinear benign reconstruction boundary is more suitable than random isolation for these mixed protocol and traffic features. Denoising discourages simple copying, while feature-error normalization prevents naturally difficult fields from dominating the score.

The recall-focused policy succeeds in detecting more than half of attacks, but it increases false alerts. Test FPR is 20.6% for Isolation Forest and 14.2% for the Autoencoder. In practice, the recall target should be chosen according to the cost of missed attacks and the investigation capacity of the security team.

Attack-family recall also varied substantially. The Autoencoder detected all Analysis records and most Generic, Worms, Exploits, and DoS records. Recall remained low for Shellcode, Fuzzers, and Reconnaissance. An operational system should therefore retain attack-family analysis rather than relying only on one overall metric.

## Key takeaways and potential improvements

UNSW-NB15 provided a stronger anomaly-detection benchmark than the previous infiltration-only daily file. The Autoencoder was the better model, but neither result demonstrates deployment readiness. Future work should evaluate repeated seeds, calibrate thresholds on more recent benign traffic, test temporally ordered data, investigate weak attack families, and examine whether a smaller selected feature set improves classical anomaly detection. Benchmark data were generated in a controlled environment, so real-network validation remains necessary.
