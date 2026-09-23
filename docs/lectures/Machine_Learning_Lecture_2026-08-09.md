# Machine Learning Lecture Notes

**Lecture date:** August 9, 2026  
**Main topic:** Regression, Linear Regression, Polynomial Regression, Model Fit, and Regression Evaluation

## 1. What Is Regression?

Regression is a statistical method used to estimate the relationship between:

- one or more **independent variables**, represented by \(X\); and
- a **dependent variable**, represented by \(Y\).

In Machine Learning terminology:

- \(X\) contains the **features**;
- \(Y\) is the **label or target**.

The model learns how \(Y\) changes when \(X\) changes and uses that relationship to predict new numerical values.

> ⭐ **Special Note:** Regression predicts a numerical value. Classification predicts a class or category.

Examples of regression predictions include:

- the amount of network traffic at a particular time;
- the number of security incidents;
- attack duration;
- investigation or remediation time; and
- a numerical risk score.

Classification questions instead include whether an email is phishing or legitimate and whether traffic is malicious or benign.

## 2. Linear Regression

Simple linear regression uses a straight-line relationship:

\[
Y = mX + c
\]

Where:

- \(X\) is the independent variable or feature;
- \(Y\) is the predicted target;
- \(m\) is the slope or coefficient; and
- \(c\) is the intercept or bias.

The slope shows the direction and strength of the relationship:

- \(m>0\): \(Y\) increases as \(X\) increases;
- \(m<0\): \(Y\) decreases as \(X\) increases;
- a larger absolute value of \(m\) indicates a steeper relationship.

The intercept represents the predicted value of \(Y\) when \(X=0\).

### Cybersecurity example

Let:

- \(X\) = number of suspicious login attempts;
- \(Y\) = number of security incidents.

A fitted regression line can estimate how many incidents may be generated when the number of suspicious logins changes.

## 3. The Best-Fit Line

A scatterplot displays the observed \((X,Y)\) points. Linear regression finds the straight line that represents the overall trend as closely as possible.

The distance between an observed value and the corresponding predicted value is the **residual**:

\[
e_i = y_i-\hat{y}_i
\]

Where:

- \(y_i\) is the actual value;
- \(\hat{y}_i\) is the predicted value.

The best-fit line aims to minimize the total prediction error.

> ⭐ **Special Note:** A line should not be selected only because it looks suitable. The residuals must be measured mathematically.

## 4. Sum of Squared Errors

The lecture introduced the **Sum of Squared Errors**, also called the residual sum of squares:

\[
SSE=\sum_{i=1}^{n}(y_i-\hat{y}_i)^2
\]

Squaring the errors:

- prevents positive and negative errors from cancelling each other;
- gives greater importance to large errors; and
- produces a single value that can be used to compare fitted models.

A smaller SSE generally indicates a line that fits the training observations more closely.

## 5. Multiple Linear Regression

When several features are used, the relationship can be written as:

\[
Y=w_0+w_1X_1+w_2X_2+\cdots+w_pX_p
\]

For example, network traffic might be predicted using:

- number of users;
- CPU usage;
- active network connections; and
- time of day.

Each coefficient represents the contribution of its corresponding feature while the other features are held constant.

## 6. Polynomial Regression

A straight line is unsuitable when the relationship follows a curve. Polynomial regression introduces powers of \(X\):

\[
Y=a_0+a_1X+a_2X^2+\cdots+a_nX^n
\]

Although the fitted curve is nonlinear with respect to \(X\), polynomial regression is still linear in its coefficients.

The highest power of \(X\) is the **degree** of the polynomial. Increasing the degree makes the curve more flexible.

### Lecture example

- \(X\) = number of suspicious logins;
- \(Y\) = investigation time.

Investigation time may increase nonlinearly as the number of suspicious events grows. A polynomial curve may therefore fit the data better than a straight line.

> ⭐ **Special Note:** A more flexible polynomial is not automatically better. A very high degree can fit noise and cause overfitting.

## 7. Underfitting, Good Fit, and Overfitting

### Underfitting

Underfitting occurs when the model is too simple to learn the underlying relationship.

Signs include:

- large errors on both training and testing data;
- a straight line applied to a strongly curved pattern;
- too few useful features.

Example: detecting phishing using only whether an email contains a URL.

### Good Fit

A well-fitted model captures the general pattern without following every small fluctuation. It balances simplicity and predictive performance.

### Overfitting

Overfitting occurs when a model learns training-data noise and unusual details instead of the general relationship.

Signs include:

- very strong training performance;
- poor performance on unseen data;
- an unnecessarily complex curve;
- too many irrelevant features.

> ⭐ **Special Note:** Feature engineering should provide enough meaningful information to prevent underfitting without adding so much noise that the model overfits.

## 8. Outliers and Data Quality

An outlier can strongly affect a regression line, particularly a polynomial curve.

Before training:

1. inspect the dataset;
2. identify missing values;
3. check data types;
4. visualize feature distributions;
5. investigate skewness and outliers;
6. examine correlations; and
7. scale or transform features when appropriate.

> ⭐ **Special Note:** Do not remove an outlier automatically. Determine whether it is an error, a legitimate rare event, or the security anomaly the model should detect.

## 9. Training and Testing

The practical demonstration divided the data into:

- **80% training data**; and
- **20% testing data**.

The training set is used to fit the coefficients. The testing set evaluates how well the fitted relationship generalizes to unseen observations.

The demonstration used `random_state=42` to make the split reproducible. The number 42 is conventional rather than mathematically required; another fixed value could be used.

## 10. Regression Evaluation Metrics

Classification measures such as accuracy, precision, recall, and F1-score are generally not the primary measures for regression.

### Mean Squared Error

\[
MSE=\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y}_i)^2
\]

A lower MSE indicates smaller squared prediction errors.

### Root Mean Squared Error

\[
RMSE=\sqrt{MSE}
\]

RMSE is expressed in the same unit as the target variable, making it easier to interpret.

### Coefficient of Determination

\[
R^2=1-\frac{\sum(y_i-\hat{y}_i)^2}{\sum(y_i-\bar{y})^2}
\]

\(R^2\) describes the proportion of target variation explained by the model. A value closer to \(1\) generally indicates a stronger fit, although it does not by itself prove that the model will generalize well.

> ⭐ **Special Note:** Compare regression models using MSE, RMSE, \(R^2\), residual behaviour, and testing performance—not classification accuracy.

## 11. Python Workflow Demonstrated

The lecture demonstrated a typical regression workflow using Pandas, NumPy, Matplotlib, Seaborn, and scikit-learn:

1. import the required libraries;
2. load or create the dataset;
3. construct a Pandas DataFrame;
4. inspect the first rows, shape, columns, data types, and summary statistics;
5. check missing values and feature distributions;
6. visualize correlations and scatterplots;
7. separate \(X\) and \(Y\);
8. create the training/testing split;
9. fit `LinearRegression`;
10. use `PolynomialFeatures` when a curved relationship is required;
11. generate predictions;
12. evaluate MSE, RMSE, and \(R^2\); and
13. plot the observations and fitted line or curve.

The lecture used both synthetic cybersecurity data and the California Housing dataset. The synthetic data produced unusually strong results because it was deliberately constructed around a clear relationship. The real dataset showed larger residuals and a lower \(R^2\).

> ⭐ **Special Note:** Excellent performance on synthetic data does not guarantee good performance on real-world data.

## 12. Dataset and Assignment Guidance

The assignment expects students to demonstrate:

- data profiling;
- missing-value checks;
- feature engineering;
- scaling or balancing where necessary;
- model training;
- testing and evaluation; and
- comparison of algorithms.

Many public datasets have already been cleaned and feature-engineered. Even when a downloaded dataset contains no missing values, the report should still show that the appropriate checks were performed.

The lecturer planned to demonstrate the phishing-detection assignment using classical and deep-learning models in a later class after resolving problems with the initial results.

> ⭐ **Assignment Note:** Do not skip profiling merely because a dataset appears clean. Show the checks and explain whether any preprocessing was required.

## Key Takeaways

- Regression predicts continuous numerical values.
- Classification predicts categories or classes.
- Linear regression fits a straight relationship.
- Polynomial regression fits curved relationships by introducing powers of the input.
- Residuals measure differences between actual and predicted values.
- The best model balances fit and simplicity.
- Too little complexity causes underfitting; excessive complexity causes overfitting.
- Regression should be evaluated with measures such as MSE, RMSE, and \(R^2\).
- Real data requires profiling, visualization, and validation before modelling.

> ⭐ **Most Important Lecture Message:** Select the regression model that matches the pattern in the data, then verify it with testing results and error measures rather than relying only on how the fitted line looks.
