# COMP70049 Machine Learning in Cyber

This repository contains lecture-derived notes, assignment planning, datasets, experiments, implementation code, and report artifacts for the COMP70049 assignment.

## Start here

- [Assignment information](./docs/assignment/ASSIGNMENT_INFORMATION.md)
- [Original assignment brief](./docs/assignment/COMP70049-Assignment.pdf)
- [Section 1 demonstration](./docs/assignment/section-01-demonstration.md)
- [Section 1 Colab notebook](./notebooks/01_phishing/section-01-email-security.ipynb)
- [Documentation index](./docs/README.md)
- [Lecture notes index](./docs/lectures/README.md)
- [Lecture note template](./docs/templates/lecture-notes-template.md)
- [Data guidance](./data/README.md)
- [Report guidance](./reports/README.md)

## Section 1 quick start

The Section 1 demonstration compares TF-IDF plus Logistic Regression with an LSTM on the assignment-recommended SpamAssassin corpus. Setup, execution, outputs, and interpretation guidance are documented in the [Section 1 walkthrough](./docs/assignment/section-01-demonstration.md).

## Repository structure

```text
.
|-- README.md                       # Repository entry point
|-- configs/                        # Experiment configuration files
|-- data/
|   |-- external/                   # Data obtained from third parties
|   |-- interim/                    # Partially transformed data
|   |-- processed/                  # Model-ready data
|   |-- raw/                        # Immutable original datasets
|   `-- samples/                    # Small, version-controlled examples
|-- docs/
|   |-- assignment/                 # Brief, extracted requirements, and planning
|   |-- decisions/                  # Dataset/model decision records
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
|   |-- figures/                    # Publication-ready figures
|   |-- final/                      # Final submission documents
|   `-- tables/                     # Generated result tables
|-- scripts/                        # Reproducible command-line workflows
|-- src/comp70049/
|   |-- common/                     # Shared preprocessing and evaluation code
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
docs/lectures/week-XX/lecture-YY-short-topic.md
```

Start each file from `docs/templates/lecture-notes-template.md`. Add every new note to the lecture index so the material remains discoverable.

### Experiments

- Use notebooks for exploration and model prototyping.
- Move reusable logic into `src/comp70049/`.
- Keep experiment parameters in `configs/` instead of hard-coding them.
- Save report-ready charts to `reports/figures/` and tables to `reports/tables/`.
- Record important dataset and model choices in `docs/decisions/`.

### Data and generated artifacts

- Never edit files in `data/raw/`; create transformed copies in `data/interim/` or `data/processed/`.
- Do not commit large datasets, trained models, secrets, or local virtual environments.
- Small anonymized samples may be committed to `data/samples/` for tests and documentation.
