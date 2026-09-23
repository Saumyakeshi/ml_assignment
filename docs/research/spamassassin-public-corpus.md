# SpamAssassin Public Corpus

## Selection rationale

The assignment brief explicitly lists the SpamAssassin Dataset as a suitable source for Section 1. The demonstration uses the official Apache public corpus because it contains raw email messages and therefore supports the required text preprocessing, TF-IDF baseline, and sequence-model workflow.

## Source

- Corpus index: <https://spamassassin.apache.org/old/publiccorpus/>
- Ham archive: `20030228_easy_ham.tar.bz2`
- Spam archive: `20030228_spam.tar.bz2`
- Accessed: 2026-09-23

The selected archives contain approximately 2,500 easy-ham messages and 500 spam messages after excluding corpus command files.

## Label definition

| Label | Meaning |
|---:|---|
| `0` | Ham, or legitimate email |
| `1` | Spam email |

## Important validity limitation

This corpus is labelled **spam versus ham**, not **phishing versus legitimate email**. It demonstrates the full text-classification process requested by the brief, but it cannot support a strong claim that the resulting model detects phishing specifically.

In the report:

- Describe the experiment as email spam detection used to demonstrate the Section 1 methodology.
- Do not relabel every spam message as phishing.
- State that a phishing-specific labelled corpus is needed before claiming phishing-detection performance.
- If time permits, repeat the same pipeline with a verified phishing-email dataset and compare domain shift.

## Processing decisions

- Parse each message using Python's standard `email` package.
- Combine the subject and readable body.
- Prefer plain-text MIME content and fall back to text extracted from HTML.
- Exclude attachments.
- Remove exact text duplicates before splitting.
- Use stratified 70% training, 15% validation, and 15% test sets.
- Learn all vocabularies and TF-IDF statistics from the training split only.

## Risks and limitations

- The corpus is old and may not represent modern email language or phishing tactics.
- Easy ham may make the classification task artificially simple.
- Spam is the minority class, requiring class-aware training and precision/recall reporting.
- Email headers may contain collection-specific artifacts that do not generalize.
- Deduplication removes exact duplicates but not near-duplicates or campaign variants.

