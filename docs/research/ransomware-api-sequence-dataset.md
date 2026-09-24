# MalwareAPI-2026 Dataset Record

## Selection

Section 4 uses the `multiclass_malware_api_seq.csv` file from
[MalwareAPI-2026](https://doi.org/10.5281/zenodo.16742661). It contains ordered
Windows API-call sequences and class labels derived from VirusTotal metadata.
The experiment retains only records labelled `Goodware` or `ransomware`, making
the target ransomware versus legitimate software.

This choice directly supports the assignment requirement to detect ransomware
from ordered system behaviour and allows the same sequence to be represented as
API-call n-grams for SVM and token indices for LSTM.

## Provenance

- Creators: Binayak Panda and Sudhanshu Shekhar Bisoyi
- Published: 3 July 2026
- Repository: Zenodo
- DOI: `10.5281/zenodo.16742661`
- Licence: Creative Commons Attribution 4.0 (`CC-BY-4.0`)
- Downloaded file: `multiclass_malware_api_seq.csv`
- File size: 68,409,365 bytes
- MD5: `4502f201b35ca1373d63182a04cdc7dc`

The source dataset contains 41,476 rows. The publisher reports that the
sequences originate from the API-call dataset released by Angelo Oliveira and
that multiclass labels were derived using VirusTotal information.

## Local profiling

| Stage | Records |
|---|---:|
| Complete multiclass CSV | 41,476 |
| Goodware and ransomware selected | 1,374 |
| Exact duplicate sequences removed | 744 |
| Unique sequences retained | 630 |
| Unique goodware sequences | 591 |
| Unique ransomware sequences | 39 |

Every retained sequence contains 100 API calls. Across the retained data there
are 210 distinct API-call names.

Exact duplicates are removed before splitting. Retaining them would allow the
same behavioral sequence to occur in training and test data, creating severe
leakage and overstating generalization.

## Experimental use

- Training: 440 unique sequences, including 27 ransomware sequences
- Validation: 95 unique sequences, including 6 ransomware sequences
- Test: 95 unique sequences, including 6 ransomware sequences
- Classical representation: TF-IDF unigrams and bigrams of API calls
- Sequential representation: integer-encoded API calls, maximum length 100

## Limitations

- Only 39 unique ransomware sequences remain after leakage-safe deduplication.
- The test set contains only six ransomware examples, so one prediction changes
  recall by approximately 16.7 percentage points.
- VirusTotal-derived labels may contain labelling noise.
- API traces were collected in controlled analysis environments and may contain
  sandbox artefacts.
- The data does not represent every ransomware family or contemporary evasion
  strategy.
- Results cannot establish production or zero-day detection performance.

## Reproduction

Open the self-contained
[Section 4 notebook](../../notebooks/04_ransomware/section-04-ransomware-detection.ipynb)
and run it from top to bottom. The notebook downloads the CSV directly from the
published Zenodo record when it is not already available.
