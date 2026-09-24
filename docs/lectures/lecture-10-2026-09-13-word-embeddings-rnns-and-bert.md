# Machine Learning — Lecture Notes

**Lecture date:** September 13, 2026  
**Main topics:** Word Embeddings, Representation Learning, Recurrent Neural Networks, Contextual Embeddings, and BERT

## 1. Lecture Overview

This lecture explained how unstructured text can be converted into numeric representations and then processed by machine-learning or deep-learning models. It introduced:

- tokenization and vocabulary construction;
- one-hot vectors and dense word embeddings;
- semantic similarity;
- Continuous Bag of Words and Skip-gram;
- Recurrent Neural Networks (RNNs);
- static versus contextual embeddings; and
- BERT/DistilBERT in a phishing-email classification example.

> ⭐ **Special Note:** A neural network cannot work directly with words as human-readable symbols. Text must first be represented numerically, but a useful representation should preserve meaning—not just assign arbitrary numbers.

---

## 2. Why Sequence Learning Is Needed

Convolutional Neural Networks are especially useful for spatial patterns, such as patterns in an image. Many datasets instead contain an important order or time relationship, including:

- sentences and documents;
- speech and audio;
- stock-market and weather data;
- sensor readings;
- network traffic; and
- sequences of images in video.

For these datasets, the current element may depend on earlier elements. Sequence-learning models are designed to preserve and use this historical context.

> ⭐ **Special Note:** The order of words changes meaning. “Dog bites man” and “man bites dog” contain the same words but describe different events.

---

## 3. From Text to Tokens

The first steps in text processing are:

1. **Tokenization:** divide the text into tokens. A token may be a word, subword, or character.
2. **Vocabulary construction:** collect the distinct tokens in the corpus.
3. **Token indexing:** assign a unique integer ID to each token.

A **corpus** is the collection of documents used as the text dataset.

The token IDs are identifiers only. Their numeric sizes do not describe meaning; for example, token ID 20 is not “greater in meaning” than token ID 5.

---

## 4. One-Hot Encoding and Its Limitations

A one-hot vector has the same dimension as the vocabulary. It contains one value of 1 and zeros everywhere else.

For a vocabulary of four words, a possible representation is:

\[
\text{cat}=[0,1,0,0]
\]

One-hot encoding uniquely identifies a token, but it has two important weaknesses:

- it produces large, sparse vectors when the vocabulary is large; and
- it contains no semantic relationship between words.

The one-hot vectors for “cat” and “dog” are no more similar than those for “cat” and “bank.”

> ⭐ **Special Note:** One-hot encoding records **identity**, while an embedding aims to record **meaning and relationships**.

---

## 5. Word Embeddings

A word embedding maps each token to a dense numeric vector:

\[
\text{token ID} \longrightarrow \mathbf{e}\in\mathbb{R}^{d}
\]

where \(d\) is the embedding dimension, chosen as a hyperparameter.

Compared with a one-hot vector, an embedding is:

- lower-dimensional;
- dense, with mostly non-zero values;
- learned from data; and
- able to place related words near each other in the embedding space.

For a sequence of \(L\) tokens, the embedding layer produces an \(L\times d\) matrix. The token order should normally be preserved when this matrix is passed to a sequence model.

### Embedding lookup table

An embedding layer contains a matrix with one row per vocabulary item and \(d\) columns. Selecting the row associated with a token ID retrieves that token’s vector. The matrix begins with random values when trained from scratch and is updated through backpropagation.

> ⭐ **Special Note:** Embeddings are representations, not final predictions. A classifier, sequence model, or generative model must still use them to perform the target task.

---

## 6. Measuring Semantic Similarity

Cosine similarity compares the directions of two embedding vectors:

\[
\cos(\theta)=
\frac{\mathbf{a}\cdot\mathbf{b}}
{\lVert\mathbf{a}\rVert\lVert\mathbf{b}\rVert}
\]

- A value close to **1** indicates similar directions and usually similar meanings.
- A value near **0** indicates little relationship.
- A negative value indicates opposing directions in the learned space.

In the practical example, related words such as “good” and “excellent” had high similarity, while positive and negative terms moved in opposing directions.

---

## 7. Learning Embeddings from Context

Words that occur in similar contexts tend to develop similar embeddings. For example, “dog” and “puppy” appear near many of the same words, so training can place them near each other in the vector space.

Embedding learning can use **self-supervision**: the text supplies its own target labels, so people do not have to label every example manually.

### 7.1 Continuous Bag of Words — CBOW

CBOW uses surrounding context words to predict a missing or target word:

\[
\text{context words} \longrightarrow \text{target word}
\]

### 7.2 Skip-gram

Skip-gram reverses the task:

\[
\text{target word} \longrightarrow \text{surrounding context words}
\]

Both tasks force the model to learn relationships among words. A prediction loss is calculated, and backpropagation updates the embedding matrix. Word2Vec is a well-known family of models built around these ideas; GloVe is another widely used static-embedding approach.

> ⭐ **Special Note:** The model does not receive a dictionary definition of each word. It learns meaning statistically from the contexts in which words appear.

---

## 8. Static Embeddings

Static embedding models assign one fixed vector to each word, regardless of where it appears. This creates a problem for words with multiple meanings:

- “I deposited money in the **bank**.”
- “He sat on the river **bank**.”

A static embedding uses the same vector for “bank” in both sentences. Another model must therefore examine the surrounding sequence to determine the intended meaning.

Pretrained static embeddings can be reused instead of training an embedding layer from scratch. This is valuable when the available task-specific dataset is small.

---

## 9. Recurrent Neural Networks — RNNs

An RNN processes a sequence one step at a time. At time \(t\), it combines the current input \(x_t\) with a hidden state from the previous step:

\[
h_t=\phi(W_xx_t+W_hh_{t-1}+b)
\]

where:

- \(h_t\) is the current hidden state;
- \(h_{t-1}\) carries information from the past;
- \(W_x\) and \(W_h\) are learned weights;
- \(b\) is a bias; and
- \(\phi\) is an activation function.

The same RNN cell and parameters are reused at every time step. Drawing the repeated computation across the sequence is called **unrolling the network through time**.

> ⭐ **Special Note:** The hidden state is the RNN’s working memory. It carries a compressed representation of earlier sequence elements into the current calculation.

### 9.1 RNN task structures

RNNs can support several input/output arrangements:

- **many-to-one:** a sequence produces one label, such as sentiment classification;
- **many-to-many:** one sequence produces another sequence, such as machine translation; and
- **one-to-many:** one input initiates a sequence of outputs.

---

## 10. Training an RNN

RNNs use **Backpropagation Through Time (BPTT)**:

1. Process the sequence and store the successive hidden states.
2. Calculate the prediction loss.
3. Propagate the error backward through the unrolled time steps.
4. Update the shared recurrent weights and other model parameters.

When an embedding layer and RNN are trained together, the same loss can update both the word vectors and the RNN parameters.

### The vanishing-gradient problem

During BPTT, many derivatives are multiplied through the chain rule. Gradients associated with distant time steps may become extremely small. The model then struggles to learn long-range relationships.

> ⭐ **Special Note:** A basic RNN often remembers recent information better than information from the distant past. LSTMs, GRUs, and Transformers were developed to handle sequence dependencies more effectively.

---

## 11. Practical Example: Sentiment Classification

The first notebook demonstrated a small binary sentiment classifier:

\[
\text{text}
\rightarrow \text{tokens}
\rightarrow \text{learned embeddings}
\rightarrow \text{SimpleRNN}
\rightarrow \text{dense output}
\rightarrow \text{positive/negative prediction}
\]

The example used:

- a vocabulary of 31 words;
- eight-dimensional embeddings;
- 16 RNN units;
- a sigmoid output for binary classification; and
- a probability threshold of 0.5.

Both the embedding vectors and RNN weights were learned during training. The embedding dimension and number of RNN units are hyperparameters; making them unnecessarily large can increase cost and overfitting.

---

## 12. Contextual Embeddings and BERT

Contextual embedding models can generate different vectors for the same token depending on its surrounding text. BERT stands for **Bidirectional Encoder Representations from Transformers**. “Bidirectional” means that it uses context from both sides of a token.

For example, the vector for “password” in a suspicious message such as “your password expires today” can differ from its vector in an ordinary security-advice message.

This provides richer information to the downstream classifier than a fixed static embedding.

> ⭐ **Special Note:** BERT supplies contextual representations, but a downstream model is still needed for tasks such as phishing detection, sentiment analysis, or multiclass classification.

Because full BERT is large, the notebook used **DistilBERT**, a smaller pretrained variant. Pretrained models are normally reused because training them from scratch requires substantial data and computing resources.

---

## 13. Practical Example: Phishing-Email Classification

The second notebook compared two classifiers that used pretrained DistilBERT embeddings:

1. **Logistic regression:** averaged the contextual token embeddings into one vector and classified it.
2. **RNN:** processed the sequence of token embeddings directly.

The sequence length was fixed at 48 tokens. Shorter sequences were padded with zeros; longer inputs would need truncation or another length-handling strategy. Each DistilBERT token vector had 768 dimensions.

On the small example dataset, logistic regression and the RNN produced similar results, but logistic regression trained much faster.

> ⭐ **Special Note:** Prefer the simplest model that meets the performance requirement. A more complex neural network is not automatically better, especially when contextual embeddings have already captured much of the useful information.

Deployment decisions should consider:

- training and inference time;
- memory and computing requirements;
- mobile or edge-device limitations;
- cloud-computing cost; and
- the risk of overfitting.

---

## 14. Key Takeaways

- Tokenization and vocabulary indexing convert raw text into identifiable units.
- One-hot vectors identify words but are sparse and do not preserve semantic similarity.
- Dense embeddings learn meaningful numeric representations from data.
- Cosine similarity measures relationships between embedding vectors.
- CBOW predicts a word from its context; Skip-gram predicts context from a word.
- Static embeddings use one vector per word, even when the meaning changes by context.
- RNNs process ordered data by carrying a hidden state from one time step to the next.
- Basic RNNs struggle with long-term dependencies because of vanishing gradients.
- Contextual models such as BERT produce different representations according to surrounding text.
- Pretrained DistilBERT embeddings can work effectively with a simple classifier.
- Model complexity should be justified by measurable improvement and deployment needs.
