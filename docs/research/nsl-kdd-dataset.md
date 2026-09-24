# NSL-KDD Dataset Record

## Selection

Section 2 uses **NSL-KDD**, one of the datasets suggested by the assignment brief. It supports multiclass network-intrusion detection with five reporting classes:

1. normal traffic;
2. denial of service (DoS);
3. probing attacks;
4. remote-to-local attacks (R2L); and
5. user-to-root attacks (U2R).

The TXT files preserve individual attack-type labels. The implementation maps those attack names to the five reporting classes rather than reducing the task to binary normal/attack detection.

## Authoritative description and mirror

The [University of New Brunswick NSL-KDD page](https://www.unb.ca/cic/datasets/nsl.html) documents the dataset, file formats, train/test protocol, improvements over KDD'99, citation requirement, and redistribution permission. UNB now states that the original download is no longer available.

The downloader therefore uses the public [HoaNP NSL-KDD mirror](https://github.com/HoaNP/NSL-KDD-DataSet), which identifies UNB as its source. The downloaded files are validated against the published partition sizes and the fixed checksums recorded below.

| File | Purpose | Rows | Columns | SHA-256 |
|---|---|---:|---:|---|
| `KDDTrain+.txt` | Official training partition with attack names and difficulty | 125,973 | 43 | `1b86d2f957b33082081bba410fe129b475efebcc13c9014c3f447c8271aadf95` |
| `KDDTest+.txt` | Official test partition with attack names and difficulty | 22,544 | 43 | `fa46b0935342616aa83b7c2578db355b6a7aaabbc492248172c7a1e8b7ab8f84` |

The 43 columns comprise 41 network-connection features, one attack-type label, and one difficulty score.

## Experimental partitioning

- The official `KDDTest+` partition remains untouched for final evaluation.
- Fifteen percent of `KDDTrain+` is selected by a reproducible stratified split for validation.
- Preprocessors and feature selection are fitted on the remaining training records only.
- The test set includes attack types not present in training, making it a harder external-distribution evaluation.

## Data quality and imbalance

NSL-KDD intentionally removed redundant training records and duplicate test records that biased the older KDD'99 benchmark. The local profiling step still checks missing values and duplicates rather than assuming the published description is sufficient.

The training data is highly imbalanced. In the current split there are only 44 U2R training examples compared with 57,241 normal and 39,038 DoS examples. Evaluation must therefore include macro-averaged metrics and per-class results rather than accuracy alone.

## Known limitations

- NSL-KDD is derived from the older KDD'99 benchmark and does not represent modern network traffic perfectly.
- Rare R2L and U2R classes make training and evaluation unstable.
- Some attack types occur only in the official test partition.
- The 41 engineered connection features are not raw packets.
- Excellent performance on this benchmark would not prove deployment readiness on a contemporary network.

These limitations must be stated in the final report.

## Reproduction

Open the self-contained [Section 2 notebook](../../notebooks/02_intrusion_detection/section-02-intrusion-detection.ipynb)
and run it from top to bottom. It includes the validated download, preprocessing,
both models, evaluation, and result export.
