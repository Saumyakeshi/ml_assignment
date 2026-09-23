# COMP70049 Machine Learning in Cyber - Assignment Information

> Source: [COMP70049-Assignment.pdf](./COMP70049-Assignment.pdf)

## Assignment at a glance

| Item | Information |
|---|---|
| Module | Machine Learning in Cyber |
| Module code | COMP70049 |
| Assignment title | Assignment |
| Contribution to module mark | 100% |
| Submission deadline | Refer to Blackboard |
| Submission location | Blackboard, using the provided link |
| Written report limit | Up to 3,000 words |
| Implementation language | Python |
| Total marks | 100 |

## Purpose and learning objectives

The assignment explores machine learning applications in cybersecurity through four areas:

1. Email security and phishing detection
2. Cyber-attack detection
3. Anomaly detection for cyber threats
4. Ransomware detection and prevention

The objectives are to:

- Understand how machine learning is applied in cybersecurity.
- Develop practical skills in data preprocessing, feature engineering, and model training.
- Compare classic machine learning and deep-learning approaches across different cybersecurity use cases.
- Evaluate models using suitable measures such as accuracy, precision, recall, and F1-score.

## Requirements for every section

Complete all of the following for each of the four sections:

1. Select an appropriate dataset, either from the examples in the brief or from another online source.
2. Preprocess the data, including cleaning, feature engineering, and normalization where appropriate.
3. Implement two models:
   - One classic machine-learning model, such as Logistic Regression, Random Forest, SVM, or Decision Tree.
   - One deep-learning model, such as LSTM, CNN, Autoencoder, or BERT.
4. Train and evaluate both models using metrics appropriate to the task.
5. Compare the models and analyze their effectiveness.
6. Include suitable visualizations, such as confusion matrices, ROC curves, precision-recall curves, and training loss/accuracy plots.

The Python implementation must be well structured, commented, executable, and free from errors.

## Section 1 - Email Security and Phishing Detection (25 marks)

### Task

Build models that detect phishing emails from their textual features.

### Suggested datasets

- Enron Spam Dataset
- SpamAssassin Dataset
- Phishing Email Dataset from the UCI Machine Learning Repository

### Preprocessing and features

- Remove stopwords and punctuation.
- Apply stemming or lemmatization.
- Convert text into numerical features with TF-IDF or word embeddings such as Word2Vec or GloVe.

### Models

- Classic ML: Logistic Regression or Random Forest
- Deep learning: LSTM or BERT

### Evaluation

- Precision
- Recall
- F1-score
- ROC curve

## Section 2 - Cyber-Attack Detection (25 marks)

### Task

Develop models that classify different types of network intrusion or cyber-attack.

### Suggested datasets

- NSL-KDD Dataset
- CIC-IDS2017 from the Canadian Institute for Cybersecurity

### Preprocessing and features

- Handle missing values.
- Standardize numerical features.
- Encode categorical variables.
- Perform feature selection or dimensionality reduction using a technique such as PCA.

### Models

- Classic ML: Decision Tree or Random Forest
- Deep learning: CNN or RNN

### Evaluation

- Accuracy
- Precision-recall curves
- Confusion matrix

## Section 3 - Anomaly Detection for Cyber Threats (25 marks)

### Task

Use unsupervised learning to identify unusual network behavior that may indicate unknown threats or zero-day attacks.

### Suggested datasets

- UNSW-NB15 Dataset
- CIC-IDS2018 anomaly and intrusion detection dataset

### Preprocessing and features

- Remove duplicate records.
- Handle missing values.
- Normalize numerical features.
- Select relevant traffic features, such as packet size, protocol type, and connection duration.

### Models

- Classic ML: Isolation Forest or One-Class SVM
- Deep learning: Autoencoder neural network

### Evaluation

- True Positive Rate (TPR)
- False Positive Rate (FPR)
- Precision-recall curves

## Section 4 - Ransomware Detection and Prevention (25 marks)

### Task

Detect ransomware activity from system behavior, including API calls and file-access patterns.

### Suggested datasets

- Windows API Call Sequence Dataset from ransomware attacks
- CIC-AndMal2017 for Android ransomware detection

### Preprocessing and features

- Convert system logs, API calls, and file-access patterns into numerical representations.
- Extract features that represent important ransomware behavior patterns.

### Models

- Classic ML: Support Vector Machine or Gradient Boosting Classifier
- Deep learning: LSTM/RNN for modeling system behavior over time

### Evaluation

- Precision
- Recall
- F1-score

## Submission deliverables

Submit the following through Blackboard:

1. A written report of no more than 3,000 words.
2. The implemented Python code or scripts.
3. A README containing instructions for running the code.

The report should cover each experiment and include:

- A description of the dataset used.
- An explanation of preprocessing and feature-engineering techniques.
- An overview of the classic ML and deep-learning models.
- A comparison of model performance.
- Relevant visualizations.
- A discussion of the findings and their cybersecurity implications.
- Key takeaways and possible improvements.

## Marking criteria

| Criterion | Description | Marks |
|---|---|---:|
| Correctness of implementation | Proper execution of the ML and deep-learning models | 50 |
| Depth of analysis | Discussion of results, insights, and model-performance comparison | 30 |
| Code quality and documentation | Readability, comments, structure, and efficiency | 10 |
| Report presentation | Clarity, organization, and completeness | 10 |
| **Total** |  | **100** |

## Completion checklist

### Planning and data

- [ ] Confirm the submission deadline on Blackboard.
- [ ] Select one suitable dataset for each section.
- [ ] Record each dataset's source and license or terms of use.
- [ ] Define the prediction target and train/validation/test strategy for each experiment.
- [ ] Avoid data leakage by fitting preprocessing steps only on training data.

### Modeling

- [ ] Implement one classic ML model for each section.
- [ ] Implement one deep-learning model for each section.
- [ ] Document preprocessing, features, architecture, and hyperparameters.
- [ ] Use reproducible random seeds where possible.
- [ ] Save the evaluation results needed for direct model comparisons.

### Evaluation and analysis

- [ ] Report the task-specific metrics listed in the brief.
- [ ] Include appropriate plots for every section.
- [ ] Compare the classic and deep-learning models fairly.
- [ ] Explain errors, limitations, security implications, and potential improvements.

### Final submission

- [ ] Keep the report within 3,000 words.
- [ ] Ensure all code is readable and commented.
- [ ] Run the full project from a clean environment and resolve all errors.
- [ ] Provide complete setup and execution instructions in the README.
- [ ] Verify that all required files are included before uploading to Blackboard.

## Important notes from the brief

- All four sections are required and carry equal marks.
- The named datasets and algorithms are presented as suitable choices; the general instructions allow an appropriate dataset introduced in the brief or sourced online.
- The brief does not state a calendar deadline; Blackboard is the authoritative source for the due date.
- The brief lists both a Python script and a file containing the implemented code among the deliverables. Treat these as one complete, runnable code submission unless Blackboard provides more specific packaging instructions.
