# Machine Learning — Lecture Notes

**Lecture date:** September 20, 2026  
**Main topics:** Natural Language Processing, Text Preprocessing, Tokenization, POS Tagging, Named-Entity Recognition, Stemming, Lemmatization, and Assignment Guidance

## 1. Lecture Overview

This lecture introduced **Natural Language Processing (NLP)** and demonstrated common text-preprocessing operations with Python NLP libraries. It also ended with guidance for the upcoming assignment submission.

The main workflow was:

\[
\text{Human language}
\rightarrow \text{raw text}
\rightarrow \text{preprocessing}
\rightarrow \text{text representation}
\rightarrow \text{ML/NLP model}
\rightarrow \text{result}
\]

> ⭐ **Special Note:** NLP is not simply recognizing individual words. The real goal is to understand or generate meaning from language in context.

---

## 2. What Is Natural Language Processing?

NLP is a branch of artificial intelligence that enables computers to process human language in forms such as:

- emails, documents, web pages, and social-media posts;
- chat messages and system logs;
- voice commands and telephone conversations;
- meeting audio; and
- interactions with virtual assistants.

The lecture divided NLP into two broad directions:

### 2.1 Natural Language Understanding — NLU

The computer interprets language produced by a person. Examples include:

- text classification;
- question answering;
- information extraction;
- named-entity recognition; and
- sentiment analysis.

### 2.2 Natural Language Generation — NLG

The computer produces human-readable language. Examples include:

- chatbot responses;
- text generation;
- summarization; and
- machine translation.

Pretrained language models such as BERT and GPT can be adapted to many of these tasks rather than always training a language model from the beginning.

---

## 3. NLP Applications

Examples discussed in the lecture included:

- **Chatbots:** interpret a request such as “I forgot my password” and direct the user to an appropriate reset process.
- **Sentiment analysis:** classify customer feedback as positive, negative, or neutral.
- **Machine translation:** convert text between languages.
- **Named-entity recognition:** identify people, organizations, locations, dates, and other important entities.
- **Text classification:** classify spam, phishing messages, news, or social-media posts.
- **Speech recognition:** convert spoken language to text for later analysis.
- **Question answering:** retrieve or generate an answer to a natural-language question.
- **Summarization:** reduce a long document to its most important points.
- **Information retrieval:** find relevant records or security incidents from a large collection.

### Cybersecurity examples

NLP can extract structured information from security text, such as:

- event type;
- IP address;
- number of failed login attempts;
- vulnerability or CVE identifier;
- affected organization; and
- possible threat context.

> ⭐ **Special Note:** Domain-specific identifiers can be essential evidence. A cleaning step that removes all numbers could accidentally delete IP addresses, CVE IDs, dates, or other security information.

---

## 4. Major NLP Challenges

Human language is difficult because meaning depends on context, culture, tone, and domain.

### 4.1 Ambiguity

The same word can have several meanings. For example, “bank” may describe a financial institution or the side of a river. The surrounding text is needed to resolve the intended meaning.

### 4.2 Sarcasm and implied meaning

“Fantastic—the server crashed again” contains a positive-looking word but expresses a negative meaning. A simple keyword system may misclassify it.

### 4.3 Idioms, metaphors, accents, and dialects

Language varies across regions and communities. The same concept can be expressed with different vocabulary, pronunciation, or figures of speech.

### 4.4 Domain-specific language

Cybersecurity, medicine, finance, and other fields contain specialized terminology and formats. A general-purpose model may need fine-tuning or carefully designed rules to interpret them reliably.

> ⭐ **Special Note:** Preprocessing alone does not solve ambiguity or sarcasm. A model also needs context, appropriate training data, and task-specific evaluation.

---

## 5. The NLP Pipeline

A typical NLP task contains three major stages:

1. **Text preprocessing:** clean and organize raw language.
2. **Text representation:** convert text into numeric features or embeddings.
3. **Model training/inference:** learn a task and produce a prediction or generated output.

The lecture concentrated on the first stage.

---

## 6. Text Cleaning

Text cleaning may remove or transform:

- punctuation;
- special characters;
- unnecessary whitespace;
- HTML or formatting artifacts;
- selected numbers; and
- noise that is irrelevant to the target task.

Regular expressions and string operations were used in the demonstration.

Cleaning must be based on the problem. Removing a phone number may be appropriate in one application but harmful in another. Likewise, punctuation can carry sentiment or mark sentence boundaries.

> ⭐ **Special Note:** Never clean text blindly. First decide which information the model needs to preserve.

---

## 7. Lowercasing and Normalization

Lowercasing converts forms such as “Apple,” “APPLE,” and “apple” to the same representation. Benefits can include:

- a smaller vocabulary;
- more consistent matching; and
- fewer duplicate forms of the same word.

However, capitalization may carry meaning. For example, “Apple” may refer to the company, while “apple” may refer to the fruit. Whether to lowercase therefore depends on the task.

The demonstration compared ordinary lowercasing with case folding, which performs a stronger Unicode-aware normalization.

---

## 8. Tokenization

Tokenization divides text into smaller units. Depending on the task, tokens may be:

- sentences;
- words;
- subwords;
- punctuation marks; or
- other meaningful patterns.

The lecture demonstrated word and sentence tokenization with common Python tools such as NLTK and spaCy, as well as simpler regular-expression patterns.

### 8.1 N-grams

An **n-gram** is a consecutive sequence of \(n\) tokens:

- **unigram:** one token, e.g. “natural”;
- **bigram:** two tokens, e.g. “natural language”;
- **trigram:** three tokens, e.g. “natural language processing.”

N-grams capture short-range context, but larger values also increase the number of possible features and make them sparser.

> ⭐ **Special Note:** Increasing \(n\) does not automatically increase accuracy. The best n-gram size depends on the dataset, task, vocabulary size, and amount of training data.

---

## 9. Part-of-Speech Tagging — POS Tagging

POS tagging assigns a grammatical role to each token, such as:

- noun;
- verb;
- adjective;
- adverb; or
- determiner.

For example, in “The cat runs quickly”:

- “The” is a determiner;
- “cat” is a noun;
- “runs” is a verb; and
- “quickly” is an adverb.

The role of a word can change with context:

- “I **book** a flight” — *book* is a verb.
- “I read a **book**” — *book* is a noun.

POS taggers may be rule-based, statistical, or based on machine/deep learning. Context-aware models are generally better able to handle ambiguous cases.

---

## 10. Named-Entity Recognition — NER

NER identifies and categorizes important entities in text, including:

- people;
- organizations;
- locations;
- dates and times;
- quantities; and
- domain-specific entities such as IP addresses.

Example:

> “On 15 September 2026, analyst Jude detected suspicious activity from 10.10.10.25 targeting Commercial Bank.”

An NER system could extract:

- **date:** 15 September 2026;
- **person:** Jude;
- **IP address:** 10.10.10.25; and
- **organization:** Commercial Bank.

> ⭐ **Special Note:** General-purpose NER models may not recognize specialized entities such as CVE IDs or IP addresses reliably. Security applications may require custom rules, training data, or fine-tuning.

---

## 11. Stop-Word Removal

Stop words are very common terms such as “the,” “is,” “a,” and “on.” Removing them can reduce noise and feature count in some tasks.

However, some apparently common words are essential to meaning:

- “I like this product.”
- “I do **not** like this product.”

If “not” is removed, the two sentences may appear to express the same sentiment.

> ⭐ **Special Note:** Do not apply a standard stop-word list without reviewing it for the task. Negations and domain terms may need to be protected.

---

## 12. Stemming and Lemmatization

Both methods reduce inflected words toward a base form, but they work differently.

### 12.1 Stemming

Stemming applies mechanical rules that remove prefixes or suffixes. It is fast but can produce forms that are not valid dictionary words.

Examples:

- “running” \(\rightarrow\) “run”;
- “studies” may become an unnatural shortened form; and
- “competition” may be reduced to a fragment.

The lecture demonstrated Porter and Snowball stemmers.

### 12.2 Lemmatization

Lemmatization uses vocabulary and grammatical information to return a valid base form called a **lemma**.

Examples:

- “running” \(\rightarrow\) “run”;
- “studies” \(\rightarrow\) “study”;
- “better” \(\rightarrow\) “good,” when the grammatical context is correctly supplied.

Lemmatization is generally more linguistically meaningful, although it is more computationally involved and still depends on accurate contextual analysis.

---

## 13. Tools Demonstrated

The practical examples used Google Colab and Python tools including:

- regular expressions;
- Python string operations;
- **NLTK** for tokenization, stop words, POS tagging, and stemming; and
- **spaCy** for tokenization, POS tagging, stop words, lemmatization, and NER.

The lecturer confirmed that **Jupyter Notebook is also acceptable** for the assignment; Colab was used mainly for convenience.

---

## 14. Assignment Guidance from the Lecture

The lecturer emphasized the quality of implementation and explanation, not only the final numeric result.

Include and explain:

- the source of each dataset;
- data profiling and whether the data was already cleaned;
- feature engineering or a justification for why it was unnecessary;
- skewness and other important distribution observations;
- multiple algorithms where the task requires comparison;
- the confusion matrix;
- accuracy, precision, recall, and F1-score; and
- why one model performed better than another.

Use the appendix for code, dataset links, notebook links, and supporting detail. The lecturer indicated that notebook and dataset links can be placed in the appendix.

> ⭐ **Special Note:** Do not paste a confusion matrix or metric values without interpreting them. Marks depend on explaining what the results mean and justifying model differences.

> ⭐ **Special Note:** If a downloaded dataset is already clean, explicitly state that and justify why additional cleaning, feature engineering, or dimensionality reduction was not required.

The lecturer said that exceeding the nominal word count was acceptable for this submission when necessary for a complete MSc-level explanation, suggesting roughly 5,000–6,000 words rather than an excessive 8,000–10,000. Because formal assessment rules take precedence, students should still check the official brief if it specifies a strict limit.

---

## 15. Key Takeaways

- NLP enables computers to understand and generate human language.
- Context is essential for resolving ambiguity, sarcasm, grammatical roles, and domain-specific meanings.
- Cleaning, lowercasing, and stop-word removal must be tailored to the task.
- Tokenization can operate at sentence, word, subword, or pattern level.
- N-grams add local context but also increase sparsity and complexity.
- POS tagging assigns grammatical roles; NER extracts important real-world entities.
- Stemming is fast and mechanical, while lemmatization aims for meaningful dictionary forms.
- Security identifiers and negations must be preserved when they affect meaning.
- In the assignment, explain the entire analytical process and interpret model results rather than presenting outputs alone.
