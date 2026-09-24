# Machine Learning — Lecture 3, Part 1 Notes

**Lecture date:** 26 July 2026  
**Main topic:** Data Pre-processing, Data Profiling, Data Distribution, and Normal Distribution

## Why Data Pre-processing is Important
Raw data often contains:
- missing values
- duplicate records
- inconsistent values
- noise
- outliers
- different scales
- unnecessary features

Goal:

\[
\boxed{
\text{Raw Data}
\rightarrow
\text{Pre-processing}
\rightarrow
\text{Clean Data}
\rightarrow
\text{ML Model}
}
\]

## Main Pre-processing Stages

\[
\boxed{
Profiling
\rightarrow
EDA
\rightarrow
Cleaning
\rightarrow
Integration
\rightarrow
Transformation
\rightarrow
Reduction
\rightarrow
Splitting
}
\]

## Data Profiling
Data profiling gives an overview of a dataset.

Check:
- rows and columns
- data types
- missing values
- unique values
- min/max
- mean/median

### ⭐ Special Note
Understand the dataset **before** building a model.

## Exploratory Data Analysis — EDA
EDA is used to understand patterns and relationships.

May include:
- distributions
- correlation
- cross-tabulation
- statistical tests
- outlier detection
- visualization

Tests mentioned:
- Chi-square
- T-test
- ANOVA

## Data Cleaning
Common tasks:
- handling missing values
- handling noise
- detecting outliers

Missing values can be treated using:
- deletion
- mean
- median
- mode
- other imputation techniques

### ⭐ Special Note
There is no one best missing-value method for every dataset.

## Noise and Outliers
Methods mentioned:
- binning
- regression
- clustering
- Z-score
- IQR

Outliers can distort statistics such as the mean.

## Data Integration
Combines data from multiple sources.

Potential problems:
- duplicate records
- conflicting values
- different units
- inconsistent naming

## Data Transformation
Methods include:
- normalization
- generalization
- aggregation

### Normalization
Transforms values into a common range, for example:

\[
0 \le x \le 1
\]

Useful when features have very different numeric scales.

## Data Reduction
Methods include:
- feature selection
- feature extraction
- sampling
- compression
- discretization

### ⭐ Special Note
More features do not automatically mean a better model.

## Data Splitting
The dataset is divided into:
- training data
- testing data

## Data Profiling in Python
The **Iris dataset** was used.

Features include:
- sepal length
- sepal width
- petal length
- petal width

Pandas can inspect data types using:

```python
df.dtypes
```

## Summary Statistics
Common statistics:
- count
- mean
- standard deviation
- minimum
- maximum
- quartiles

Mean:

\[
\text{Mean} =
\frac{x_1+x_2+\cdots+x_n}{n}
\]

## Cardinality

\[
\boxed{\text{Cardinality} = \text{number of unique values}}
\]

### Low Cardinality
Few unique values.

Example:
- Male
- Female

### High Cardinality
Many unique values.

Examples:
- User ID
- Email
- Transaction ID
- IP address

### ⭐ Special Note
High-cardinality identifiers may add complexity without helping prediction.

## Data Distribution
A distribution describes how values are spread.

Examples:
- Normal
- Uniform
- Skewed

## Normal Distribution
The Normal (Gaussian) Distribution is a symmetric bell-shaped distribution.

For an ideal normal distribution:

\[
\text{Mean}=\text{Median}=\text{Mode}
\]

Probability density:

\[
f(x)=
\frac{1}{\sigma\sqrt{2\pi}}
e^{-\frac{1}{2}\left(\frac{x-\mu}{\sigma}\right)^2}
\]

where:
- \(\mu\) = mean
- \(\sigma\) = standard deviation
- \(\sigma^2\) = variance

### Effect of Mean
\[
\boxed{\mu \rightarrow \text{center/location}}
\]

### Effect of Standard Deviation
\[
\boxed{\sigma \rightarrow \text{spread}}
\]

## Z-Score

\[
\boxed{
z=\frac{x-\mu}{\sigma}
}
\]

The Z-score tells how many standard deviations a value lies from the mean and can help detect outliers.

# ⭐ Important Revision Points
1. Preprocess raw data before training
2. Profiling → EDA → Cleaning → Integration → Transformation → Reduction → Splitting
3. Missing-value handling
4. Outlier detection
5. Normalization
6. Feature reduction
7. Cardinality
8. High-cardinality IDs may be unhelpful
9. Normal distribution
10. \(\mu\) = center
11. \(\sigma\) = spread
12. Z-score for distance from the mean
