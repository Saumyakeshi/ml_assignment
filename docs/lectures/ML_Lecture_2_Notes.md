# Machine Learning — Lecture 2 Notes

**Lecture date:** 19 July 2026  
**Main focus:** Google Colab, Python basics, datasets, features and labels, Decision Tree classification, and model evaluation.

## Google Colab
Google Colab is a cloud-based Jupyter Notebook environment for running Python in a browser.

Benefits:
- no local installation
- notebooks stored online
- Python libraries available
- notebooks can be shared
- GPU/TPU support may be available

### ⭐ Special Note
Basic Python and Google Colab are important practical tools for this module.

## Basic Python Data Types

```python
age = 25
height = 5.9
name = "Alice"
is_student = True
```

Types:
- Integer
- Float
- String
- Boolean

## Working with Pandas
The lecture used Pandas DataFrames to represent tabular datasets.

Example:

| Packets | LoginAttempts | FailedLogins | Label |
|---:|---:|---:|---:|
| 120 | 1 | 0 | 0 |
| 300 | 10 | 8 | 1 |
| 50 | 0 | 0 | 0 |
| 500 | 15 | 10 | 1 |

Rows are observations and columns are variables/features.

## Features and Labels

\[
X = \text{features}
\]

\[
y = \text{target/label}
\]

For the cybersecurity example:
- Packets
- LoginAttempts
- FailedLogins

were features.

The label represented:

\[
0 = \text{Normal}
\]

\[
1 = \text{Attack}
\]

### ⭐ Special Note
\[
\boxed{X=\text{Inputs},\qquad y=\text{Correct Output}}
\]

## Cybersecurity Classification
The model classified activity as:
- normal
- attack

This is a **binary classification** problem.

## Train/Test Split
The dataset is split into:
- training data
- testing data

Training data teaches the model. Testing data evaluates performance on unseen examples.

## Decision Tree Classifier
A Decision Tree learns rules from input features.

Typical workflow:

\[
\text{Create Model}
\rightarrow
\text{Fit}
\rightarrow
\text{Predict}
\]

## Accuracy

\[
\text{Accuracy}
=
\frac{\text{Correct Predictions}}
{\text{Total Predictions}}
\]

### ⭐ Important Caution
100% accuracy on a tiny dataset does not prove the model is good in the real world.

## Confusion Matrix

| | Predicted Positive | Predicted Negative |
|---|---:|---:|
| Actual Positive | True Positive | False Negative |
| Actual Negative | False Positive | True Negative |

- **TP:** attack predicted and actually attack
- **TN:** normal predicted and actually normal
- **FP:** attack predicted but actually normal
- **FN:** normal predicted but actually attack

### ⭐ Special Note
In cybersecurity, **False Negatives can be especially serious** because attacks are missed.

## Precision

\[
\boxed{Precision=\frac{TP}{TP+FP}}
\]

Question:
> When the model says “attack”, how often is it correct?

## Recall

\[
\boxed{Recall=\frac{TP}{TP+FN}}
\]

Question:
> Of all actual attacks, how many did the model detect?

## F1-Score

\[
\boxed{
F1 =
2\frac{Precision\times Recall}{Precision+Recall}
}
\]

Balances precision and recall.

## Classification Report
Includes:
- precision
- recall
- F1-score
- support

**Support** = number of true samples in a class.

## Correlation

\[
-1 \le r \le 1
\]

- \(r\approx1\): strong positive relationship
- \(r\approx0\): little/no linear relationship
- \(r\approx-1\): strong negative relationship

### ⭐ Special Note
\[
\boxed{\text{Correlation does not imply causation}}
\]

## Typical ML Workflow

\[
\boxed{
\text{Data}
\rightarrow
\text{Features/Labels}
\rightarrow
\text{Train/Test Split}
\rightarrow
\text{Training}
\rightarrow
\text{Prediction}
\rightarrow
\text{Evaluation}
}
\]

## Real Cybersecurity Datasets
Real datasets may contain:
- normal traffic
- malicious traffic
- attack categories
- flow/packet information
- login behavior
- network statistics

Challenges may include:
- missing values
- noisy data
- class imbalance
- redundant features
- many samples and feature types

# ⭐ Important Revision Points
1. Google Colab and basic Python
2. \(X\) = features, \(y\) = label
3. Binary classification
4. Decision Trees
5. Train/test split
6. Accuracy
7. TP, TN, FP, FN
8. Precision
9. Recall
10. F1-score
11. Accuracy alone is not enough
12. Correlation does not imply causation
