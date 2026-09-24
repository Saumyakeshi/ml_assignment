# Machine Learning Lecture Notes

**Lecture date:** August 16, 2026  
**Main topic:** Logistic Regression, Model Training, K-Nearest Neighbours, Cross-Validation, and Classification Evaluation

## 1. Lecture Overview

The lecture introduced two supervised classification algorithms:

- **Logistic Regression**
- **K-Nearest Neighbours (KNN)**

Both learn from labelled examples, where the features are inputs and the label is the expected class.

The lecturer also recommended two research papers supplied with the Week 5 material:

- port-scan detection using Logistic Regression and other ML models;
- black-hole attack detection using KNN.

These papers demonstrate the type of algorithm comparison expected in the assignment.

## 2. Logistic Regression

Despite its name, Logistic Regression is primarily a **classification** algorithm. In its basic form, it predicts one of two classes.

Examples:

- phishing or legitimate;
- attack or normal;
- disease or no disease;
- pass or fail;
- purchase or no purchase.

The model first calculates a weighted score:

\[
z=b+w_1x_1+w_2x_2+\cdots+w_px_p
\]

Where:

- \(x_i\) is a feature;
- \(w_i\) is its learned weight;
- \(b\) is the bias or intercept.

For phishing detection, features could include:

- suspicious URLs;
- malicious attachments;
- urgent language;
- sender reputation;
- domain characteristics.

Features that are more useful to the prediction should receive stronger learned weights.

> ⭐ **Special Note:** Linear Regression predicts a continuous number. Logistic Regression converts a score into a class probability and then predicts a category.

## 3. Sigmoid Function

The weighted score \(z\) can be any real number. The sigmoid function converts it to a probability between \(0\) and \(1\):

\[
\sigma(z)=\frac{1}{1+e^{-z}}
\]

For example, when \(z=2\):

\[
\sigma(2)\approx0.88
\]

This can be interpreted as an estimated probability of approximately 88% for the positive class.

A common classification threshold is \(0.5\):

- probability \(\geq0.5\): predict class \(1\);
- probability \(<0.5\): predict class \(0\).

> ⭐ **Technical Clarification:** The classification threshold is not a statistical **p-value**. It is a decision boundary that can be adjusted depending on the cost of false positives and false negatives.

> ⭐ **Special Note:** In security applications, \(0.5\) is not always the best threshold. A lower threshold may detect more attacks but also create more false alarms.

## 4. Training Logistic Regression

Training adjusts the weights and bias so that predicted probabilities become closer to the correct labels.

The general process is:

1. initialize the weights and bias;
2. calculate \(z\);
3. apply the sigmoid function;
4. compare the prediction with the true label;
5. calculate the loss;
6. calculate the gradient;
7. update the weights;
8. repeat until the model converges or reaches the iteration limit.

### Binary Cross-Entropy Loss

The standard Logistic Regression loss is:

\[
L=-\frac{1}{n}\sum_{i=1}^{n}
\left[y_i\log(\hat{y}_i)+(1-y_i)\log(1-\hat{y}_i)\right]
\]

Where:

- \(y_i\) is the actual class;
- \(\hat{y}_i\) is the predicted probability.

Confident wrong predictions produce a large loss.

### Gradient Descent

Gradient descent updates the model parameters in the direction that reduces loss:

\[
w_{new}=w_{old}-\alpha\frac{\partial L}{\partial w}
\]

The learning rate \(\alpha\) controls the size of each update.

- Too large: training may overshoot the minimum.
- Too small: training may be unnecessarily slow.

> ⭐ **Special Note:** Training is not simply memorizing fixed feature weights. The model repeatedly adjusts them to reduce prediction error.

## 5. Logistic Regression Assumptions and Limitations

Logistic Regression works best when:

- observations are reasonably independent;
- features are not unnecessarily duplicated or highly correlated;
- continuous features have an approximately linear relationship with the **log-odds**;
- the dataset contains enough examples for the parameters being estimated;
- influential outliers are investigated;
- useful features have been selected and prepared.

Limitations include:

- difficulty learning strongly nonlinear decision boundaries;
- sensitivity to poor feature engineering and multicollinearity;
- weaker performance when classes overlap heavily;
- misleading results on badly imbalanced data.

## 6. Extensions of Logistic Regression

### Multinomial Logistic Regression

Used when the target has more than two possible classes, such as:

- normal traffic;
- DDoS;
- SQL injection;
- port scanning;
- brute force.

### Softmax Function

Softmax converts several class scores into probabilities that add up to one. The class with the largest probability is normally selected.

### Regularization

Regularization discourages unnecessarily large coefficients and can reduce overfitting.

- **L1 regularization:** can reduce some coefficients to zero and perform feature selection.
- **L2 regularization:** shrinks coefficients while usually retaining all features.

> ⭐ **Special Note:** Regularization is different from dimensionality reduction, although both may reduce the effect of unnecessary features.

## 7. K-Nearest Neighbours

KNN is a supervised, instance-based algorithm. It classifies a new observation using the labels of nearby training observations.

For classification:

1. select a value for \(K\);
2. calculate the distance from the new point to the training points;
3. identify the \(K\) closest neighbours;
4. assign the most common class among them.

For regression, KNN can use the average target value of the nearest neighbours.

> ⭐ **Special Note:** KNN is often called a lazy learner because it stores the training examples and performs most of its computation when a prediction is requested.

## 8. Distance Measures

### Euclidean Distance

\[
d(\mathbf{x},\mathbf{y})=
\sqrt{\sum_{j=1}^{p}(x_j-y_j)^2}
\]

For two dimensions:

\[
d=\sqrt{(x_1-x_2)^2+(y_1-y_2)^2}
\]

### Manhattan Distance

\[
d(\mathbf{x},\mathbf{y})=
\sum_{j=1}^{p}|x_j-y_j|
\]

The chosen measure defines what “near” means to the model.

## 9. Selecting the Value of K

The value of \(K\) controls the model’s complexity.

- **Small \(K\):** sensitive to noise and outliers; may overfit.
- **Large \(K\):** produces a smoother boundary but may include unrelated observations and underfit.

The best value should be selected using validation results rather than assumption.

### K-Fold Cross-Validation

Cross-validation divides the training data into several folds. The model is trained on most folds and validated on the remaining fold. This is repeated until every fold has served as the validation set.

For each candidate \(K\):

1. run cross-validation;
2. average the validation performance;
3. compare error or accuracy across candidate values;
4. select a value that generalizes well.

> ⭐ **Special Note:** A larger \(K\) is not automatically more accurate. The best value depends on the dataset.

## 10. Why Scaling Is Essential for KNN

KNN is distance-based, so features with large numerical ranges can dominate the calculation.

Example:

- connection duration: 0–60;
- packet count: 0–1,000,000.

Without scaling, packet count may control the distance even when failed connections or scanned ports are more important.

Standardization is commonly applied:

\[
z=\frac{x-\mu}{\sigma}
\]

> ⭐ **Special Note:** Fit the scaler using the training data, then apply that fitted transformation to both training and testing data. Do not fit the scaler separately on the test set.

## 11. KNN Performance and Optimisation

KNN can become expensive because it may compare a new point with many stored observations.

Possible improvements include:

- feature selection;
- dimensionality reduction;
- removing redundant features;
- KD-trees;
- Ball Trees;
- approximate nearest-neighbour methods for very large datasets.

KNN can also suffer from the curse of dimensionality: as the number of features grows, distances become less informative.

## 12. Practical Demonstration

The lecture demonstrated KNN and Logistic Regression with synthetic network-security data containing features such as:

- packet count;
- connection duration;
- bytes sent and received;
- failed connections or failed logins;
- scanned or unique ports;
- attack/normal target label.

The workflow was:

1. create or load a DataFrame;
2. verify the class distribution;
3. separate features \(X\) from target \(Y\);
4. split into 80% training and 20% testing;
5. fit a `StandardScaler` on the training features;
6. train KNN or Logistic Regression;
7. generate predictions;
8. examine accuracy, precision, recall, F1-score, and the confusion matrix;
9. test several values of \(K\);
10. test new network observations.

The perfectly separated synthetic dataset produced an accuracy of 1.0 for many values of \(K\). A second, noisier dataset produced more realistic performance near 0.96–0.97.

> ⭐ **Special Note:** Perfect performance on synthetic data can indicate that the generated classes are too easy to separate. It does not demonstrate real-world reliability.

The lecturer also observed that a model with high reported accuracy still made an implausible prediction for a manually entered example. This illustrates why metrics, data quality, and individual error analysis must all be checked.

## 13. Classification Evaluation

Important measures include:

- **Accuracy:** proportion of all predictions that are correct;
- **Precision:** proportion of predicted attacks that are genuine attacks;
- **Recall:** proportion of genuine attacks that were detected;
- **F1-score:** harmonic mean of precision and recall;
- **Confusion matrix:** true positives, true negatives, false positives, and false negatives;
- **ROC-AUC:** ability to rank positive examples above negative examples across thresholds.

> ⭐ **Special Note:** Accuracy alone is insufficient, especially when one class is much more common than the other.

## 14. Phishing Dataset and Assignment Guidance

The phishing-email practical used a dataset with approximately 112 columns. Logistic Regression and Random Forest were compared using classification metrics and ROC-AUC. Random Forest performed slightly better for that particular dataset.

The class counts discussed in the lecture were roughly 8,000 legitimate emails and 1,037 phishing emails. This is an imbalanced dataset and should be handled and evaluated accordingly.

Assignment expectations include:

- data preprocessing and feature engineering;
- comparison of classical models such as Logistic Regression and Random Forest;
- later comparison with Deep Learning models such as LSTM or BERT;
- reporting accuracy, precision, recall, F1-score, confusion matrices, and related measures.

The lecturer planned to demonstrate two of the four assignment scenarios; students would complete the remaining two using the same workflow.

> ⭐ **Assignment Note:** Show evidence of data cleaning and feature engineering. Do not rely only on a pre-cleaned dataset without explaining the checks performed.

## Key Takeaways

- Logistic Regression maps a weighted score to a probability using the sigmoid function.
- A decision threshold converts the probability into a class.
- Cross-entropy loss and gradient descent are used to learn the coefficients.
- KNN classifies a point using the classes of nearby observations.
- The value of \(K\) should be selected through validation.
- Scaling is essential for distance-based algorithms.
- Accuracy must be supported by precision, recall, F1, confusion-matrix analysis, and testing on realistic data.
- High performance on synthetic data may not transfer to real security data.

> ⭐ **Most Important Lecture Message:** Understand how each algorithm makes its decision so that model selection is based on the data and problem, not merely on whichever implementation gives the highest headline accuracy.
