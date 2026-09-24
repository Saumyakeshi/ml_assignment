# Machine Learning — Lecture 1 Notes

**Lecture date:** 12 July 2026  
**Topic:** Introduction to Machine Learning

## Summary
This lecture introduced the basic concepts of Machine Learning: learning patterns from data, supervised and unsupervised learning, regression vs classification, training vs testing, generalization, underfitting, and overfitting.

## What is Machine Learning?
Machine Learning allows a computer to learn patterns from examples or data rather than explicitly programming every rule.

\[
y=f(x)
\]

- \(x\): input/features
- \(f\): learned model
- \(y\): output/prediction

## Main Types of Machine Learning

### Supervised Learning
Training data contains both input and the correct output.

Examples:
- Classification
- Regression

### Unsupervised Learning
Training data has no known labels. The algorithm tries to discover groups, similarities, or hidden patterns.

### ⭐ Special Note
**Supervised = labels are provided.**  
**Unsupervised = labels are not provided.**

## Regression vs Classification

### Regression
Predicts continuous numerical values.

Examples:
- house price
- temperature
- sales

### Classification
Predicts a discrete category/class.

Examples:
- Spam / Not Spam
- Malicious / Benign
- Fraud / Legitimate

### ⭐ Special Note
If the output is a **continuous number** → Regression.  
If the output is a **category/class** → Classification.

## Classification Examples
The lecture used examples such as:
- spam detection
- face recognition
- nearest-neighbor classification
- linear classification

### Nearest Neighbor
A new point is classified using the class of nearby labelled examples.

### Linear Classifier
A linear classifier attempts to find a decision boundary separating classes.

\[
h(x)=\operatorname{sign}(w^Tx+b)
\]

## Training vs Testing

### Training Data
Used to learn/build the model.

### Test Data
Used to evaluate the model using unseen examples.

### ⭐ Special Note
Good performance on training data alone is not enough.

## Generalization
Generalization is the ability of a model to perform well on data it has not seen before.

\[
\boxed{\text{Good ML model} \Rightarrow \text{Good performance on unseen data}}
\]

## Underfitting
The model is too simple.

Typical characteristics:
- high training error
- high test error
- high bias

## Overfitting
The model fits the training data too closely.

Typical characteristics:
- very low training error
- higher test error
- high variance

| Model | Training Performance | Test Performance |
|---|---|---|
| Underfitting | Poor | Poor |
| Good model | Good | Good |
| Overfitting | Very good | Poorer |

### ⭐ Special Note
The goal is not merely to minimize training error. The model should **generalize well**.

## Typical ML Workflow

\[
\text{Data}
\rightarrow
\text{Prepare}
\rightarrow
\text{Train}
\rightarrow
\text{Predict}
\rightarrow
\text{Evaluate}
\]

# ⭐ Important Revision Points
1. \(y=f(x)\)
2. Supervised vs unsupervised learning
3. Classification vs regression
4. Training vs testing
5. Generalization
6. Underfitting = too simple / high bias
7. Overfitting = memorizes training data / high variance
8. Nearest-neighbor and linear classification basics

## Very Short Revision Summary
- **Supervised:** input + label
- **Unsupervised:** input only
- **Regression:** continuous output
- **Classification:** category
- **Training:** learn model
- **Testing:** evaluate unseen data
- **Generalization:** perform well on unseen data
