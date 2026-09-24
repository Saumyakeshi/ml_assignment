# COMP70049 Machine Learning in Cyber

This repository contains lecture-derived notes, assignment planning, datasets, experiments, implementation code, and report artifacts for the COMP70049 assignment.

## Start here

- [Assignment information](./docs/assignment/ASSIGNMENT_INFORMATION.md)
- [Original assignment brief](./docs/assignment/COMP70049-Assignment.pdf)
- [Assignment guidance extracted from lectures](./docs/assignment/LECTURE_ASSIGNMENT_GUIDANCE.md)
- [Section 1 demonstration](./docs/assignment/section-01-demonstration.md)
- [Section 1 Colab notebook](./notebooks/01_phishing/section-01-email-security.ipynb)
- [Section 2 demonstration](./docs/assignment/section-02-demonstration.md)
- [Section 2 Colab notebook](./notebooks/02_intrusion_detection/section-02-intrusion-detection.ipynb)
- [Section 3 demonstration](./docs/assignment/section-03-demonstration.md)
- [Section 3 Colab notebook](./notebooks/03_anomaly_detection/section-03-anomaly-detection.ipynb)
- [Documentation index](./docs/README.md)
- [Lecture notes index](./docs/lectures/README.md)
- [Lecture note template](./docs/templates/lecture-notes-template.md)
- [Data guidance](./data/README.md)
- [Report guidance](./reports/README.md)

## Section 1 quick start

The Section 1 demonstration compares TF-IDF plus Logistic Regression with an LSTM on the assignment-recommended SpamAssassin corpus. Setup, execution, outputs, and interpretation guidance are documented in the [Section 1 walkthrough](./docs/assignment/section-01-demonstration.md).

## Section 2 quick start

The Section 2 demonstration performs five-class intrusion detection on NSL-KDD, comparing a class-balanced Random Forest with a class-weighted 1D CNN. See the [Section 2 walkthrough](./docs/assignment/section-02-demonstration.md).

## Section 3 quick start

The Section 3 demonstration cleans an unmodified CSE-CIC-IDS2018 daily flow CSV and compares a normal-only Isolation Forest with an Autoencoder. See the [Section 3 walkthrough](./docs/assignment/section-03-demonstration.md).

## Repository structure

```text
.
|-- README.md                       # Repository entry point
|-- configs/                        # Experiment configuration files
|-- data/
|   |-- processed/                  # Model-ready manifests and outputs
|   `-- raw/                        # Immutable original datasets
|-- docs/
|   |-- assignment/                 # Brief, extracted requirements, and planning
|   |-- lectures/                   # Notes generated from lecture videos
|   |-- research/                   # Papers, links, and literature notes
|   `-- templates/                  # Reusable Markdown templates
|-- models/                         # Generated model artifacts (not committed)
|-- notebooks/
|   |-- 01_phishing/                # Section 1 exploration and experiments
|   |-- 02_intrusion_detection/     # Section 2 exploration and experiments
|   |-- 03_anomaly_detection/       # Section 3 exploration and experiments
|   `-- 04_ransomware/              # Section 4 exploration and experiments
|-- reports/
|   |-- drafts/                     # Report source and working drafts
|   |-- section_01/                 # Section 1 metrics, tables, and figures
|   |-- section_02/                 # Section 2 metrics, tables, and figures
|   `-- section_03/                 # Section 3 metrics, tables, and figures
|-- scripts/                        # Reproducible command-line workflows
|-- src/comp70049/
|   |-- phishing/                   # Section 1 implementation
|   |-- intrusion_detection/        # Section 2 implementation
|   |-- anomaly_detection/          # Section 3 implementation
|   `-- ransomware/                 # Section 4 implementation
`-- tests/                          # Automated tests
```

## Working conventions

### Lecture notes

Store notes under `docs/lectures/` using:

```text
docs/lectures/lecture-NN-YYYY-MM-DD-short-topic.md
```

Start each file from `docs/templates/lecture-notes-template.md`. Add every new note to the lecture index so the material remains discoverable.

### Experiments

- Use notebooks for exploration and model prototyping.
- Move reusable logic into `src/comp70049/`.
- Keep experiment parameters in `configs/` instead of hard-coding them.
- Save metrics, tables, and report-ready charts under `reports/section_NN/`.
- Record important dataset and model choices in the relevant assignment or research document.

### Data and generated artifacts

- Never edit files in `data/raw/`; create model-ready manifests or transformed outputs in `data/processed/`.
- Do not commit large datasets, trained models, secrets, or local virtual environments.
