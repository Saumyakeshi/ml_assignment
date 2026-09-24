# Machine Learning — Lecture 3 Notes

**Lecture date:** 26 July 2026  
**Main topics:** Data preprocessing, data profiling, normality, skewness, and probability distributions

## Part 1 — Data Preprocessing and Profiling

### Why Data Pre-processing is Important
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

### Main Pre-processing Stages

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

### Data Profiling
Data profiling gives an overview of a dataset.

Check:
- rows and columns
- data types
- missing values
- unique values
- min/max
- mean/median

#### ⭐ Special Note
Understand the dataset **before** building a model.

### Exploratory Data Analysis — EDA
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

### Data Cleaning
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

#### ⭐ Special Note
There is no one best missing-value method for every dataset.

### Noise and Outliers
Methods mentioned:
- binning
- regression
- clustering
- Z-score
- IQR

Outliers can distort statistics such as the mean.

### Data Integration
Combines data from multiple sources.

Potential problems:
- duplicate records
- conflicting values
- different units
- inconsistent naming

### Data Transformation
Methods include:
- normalization
- generalization
- aggregation

#### Normalization
Transforms values into a common range, for example:

\[
0 \le x \le 1
\]

Useful when features have very different numeric scales.

### Data Reduction
Methods include:
- feature selection
- feature extraction
- sampling
- compression
- discretization

#### ⭐ Special Note
More features do not automatically mean a better model.

### Data Splitting
The dataset is divided into:
- training data
- testing data

### Data Profiling in Python
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

### Summary Statistics
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

### Cardinality

\[
\boxed{\text{Cardinality} = \text{number of unique values}}
\]

#### Low Cardinality
Few unique values.

Example:
- Male
- Female

#### High Cardinality
Many unique values.

Examples:
- User ID
- Email
- Transaction ID
- IP address

#### ⭐ Special Note
High-cardinality identifiers may add complexity without helping prediction.

### Data Distribution
A distribution describes how values are spread.

Examples:
- Normal
- Uniform
- Skewed

### Normal Distribution
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

#### Effect of Mean
\[
\boxed{\mu \rightarrow \text{center/location}}
\]

#### Effect of Standard Deviation
\[
\boxed{\sigma \rightarrow \text{spread}}
\]

### Z-Score

\[
\boxed{
z=\frac{x-\mu}{\sigma}
}
\]

The Z-score tells how many standard deviations a value lies from the mean and can help detect outliers.

### ⭐ Important Revision Points
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

---

## Part 2 — Probability Distributions

### Continuation from Part 1

Part 1 introduced data preprocessing, data profiling, distributions, the normal distribution, standard deviation, and Z-scores.

Part 2 continues mainly with **probability distributions** and how to identify different types of data distributions.

A probability distribution describes:

\[
\boxed{\text{how likely different values are to occur}}
\]

Understanding the distribution of a feature is important because many statistical and Machine Learning methods make assumptions about the data.

---

### Normal Distribution Recap

A **Normal Distribution** is a continuous probability distribution with a bell-shaped curve.

It is controlled mainly by:

\[
\mu = \text{Mean}
\]

and

\[
\sigma = \text{Standard Deviation}
\]

The Probability Density Function is:

\[
f(x)=
\frac{1}{\sigma\sqrt{2\pi}}
e^{-\frac{1}{2}\left(\frac{x-\mu}{\sigma}\right)^2}
\]

For an ideal normal distribution:

\[
\boxed{\text{Mean}=\text{Median}=\text{Mode}}
\]

and the curve is symmetrical around the mean.

---

### Effect of Mean and Standard Deviation

Changing the **mean** moves the center of the distribution.

\[
\boxed{\mu \rightarrow \text{location}}
\]

Changing the **standard deviation** changes the spread.

\[
\boxed{\sigma \rightarrow \text{spread}}
\]

A smaller standard deviation gives a narrower distribution, while a larger standard deviation gives a wider one.

#### ⭐ Special Note

Two normal distributions can have:

- the same mean but different spreads, or
- different means but similar spreads.

Therefore, both \(\mu\) and \(\sigma\) are needed to describe a normal distribution.

---

### Skewness

Real-world data does not always follow a perfectly symmetric normal distribution.

A distribution can be **skewed**.

#### Positive Skew / Right Skew

A positively skewed distribution has a longer tail on the **right side**.

```text
   /\
  /  \
 /    \______
```

Most observations occur toward the lower side, while a smaller number of large values extend the tail to the right.

Typical relationship:

\[
\text{Mean} > \text{Median}
\]

Examples can include:

- income
- response time
- network latency
- file sizes

#### Negative Skew / Left Skew

A negatively skewed distribution has a longer tail toward the **left side**.

```text
______ 
      \ 
       \ /\
```

Typical relationship:

\[
\text{Mean} < \text{Median}
\]

#### ⭐ Special Note

Remember the direction of skewness using the **tail**, not where most of the values are.

\[
\boxed{\text{Right tail} \Rightarrow \text{Positive skew}}
\]

\[
\boxed{\text{Left tail} \Rightarrow \text{Negative skew}}
\]

---

### Why Skewness Matters

Many ML and statistical techniques work better when the data is reasonably well behaved.

Strongly skewed data can:

- affect the mean
- affect standard deviation
- make outlier detection difficult
- violate assumptions of statistical tests
- affect some ML models

Therefore, checking data distribution is part of preprocessing.

Possible transformations for skewed data can include:

- logarithmic transformation
- square-root transformation
- other scaling techniques

depending on the dataset.

---

### Histogram and KDE

The lecture used visual methods to inspect distribution shape.

#### Histogram

A histogram divides values into ranges called **bins** and shows how many observations occur in each range.

It helps identify:

- symmetry
- skewness
- outliers
- multiple peaks

#### KDE — Kernel Density Estimate

A **Kernel Density Estimate** produces a smoother estimate of the distribution.

A histogram may look like discrete bars, while KDE gives a smooth curve.

#### ⭐ Special Note

Visual inspection is useful, but it should not be the only way to determine whether data follows a normal distribution.

Statistical tests can also be used.

---

### Testing for Normality

The lecture demonstrated methods for checking whether a dataset is normally distributed.

Two tests shown were:

- **Shapiro-Wilk Test**
- **Kolmogorov-Smirnov Test**

---

### Shapiro-Wilk Test

The Shapiro-Wilk test checks whether the data is consistent with a normal distribution.

The hypotheses can be interpreted as:

\[
H_0: \text{Data follows a normal distribution}
\]

\[
H_1: \text{Data does not follow a normal distribution}
\]

The result provides a **p-value**.

A commonly used threshold is:

\[
\alpha = 0.05
\]

If:

\[
p > 0.05
\]

there is not enough evidence to reject normality.

If:

\[
p < 0.05
\]

the data is usually considered significantly different from a normal distribution.

#### ⭐ Special Note

A useful rule for revision:

\[
\boxed{p>0.05 \Rightarrow \text{normality is plausible}}
\]

\[
\boxed{p<0.05 \Rightarrow \text{evidence against normality}}
\]

This does not prove that a dataset is perfectly normal; it is a statistical decision based on the test.

---

### Kolmogorov-Smirnov Test

The **Kolmogorov-Smirnov test** can compare the observed data distribution against a reference distribution.

For normality checking, it can be used to compare a sample with a normal distribution.

As with many hypothesis tests, interpretation is based on the p-value.

The general principle remains:

\[
\text{Compare } p \text{ with significance level } \alpha
\]

---

### Uniform Distribution

The lecture then introduced the **Uniform Distribution**.

In a uniform distribution, every value within a specified range has an equal probability.

If:

\[
a < x < b
\]

the probability density is:

\[
\boxed{
f(x)=\frac{1}{b-a}
}
\]

within that interval.

Outside the interval:

\[
f(x)=0
\]

Graphically:

```text
       ___________
      |           |
______|           |______
      a           b
```

The graph is flat because each value in the interval is equally likely.

---

### Uniform Distribution Example

One example discussed was **random cryptographic key generation**.

Ideally, keys should be generated so that every possible valid key has an equal probability of being selected.

A uniform distribution is suitable for this because no one outcome should be favored.

#### ⭐ Special Note

Uniform distribution means:

\[
\boxed{\text{All outcomes in the allowed range have equal probability}}
\]

This is particularly important in security-related random generation.

---

### Continuous vs Discrete Distributions

The lecture covered both continuous and discrete distributions.

#### Continuous distribution

Can take any value within an interval.

Examples:

- Normal
- Uniform
- Exponential

#### Discrete distribution

Takes countable values such as:

\[
0,1,2,3,\ldots
\]

Examples:

- Binomial
- Poisson

#### ⭐ Special Note

A useful distinction:

\[
\boxed{\text{Continuous} \rightarrow \text{measurements}}
\]

\[
\boxed{\text{Discrete} \rightarrow \text{counts/outcomes}}
\]

---

### Binomial Distribution

A **Binomial Distribution** models the number of successes in a fixed number of independent trials.

Each trial has only two possible outcomes, such as:

- success / failure
- attack / no attack
- correct / incorrect
- malicious / benign

The probability of exactly \(k\) successes is:

\[
\boxed{
P(X=k)
=
{n \choose k}
p^k(1-p)^{n-k}
}
\]

where:

- \(n\) = total number of trials
- \(k\) = number of successes
- \(p\) = probability of success
- \(1-p\) = probability of failure

---

### Conditions for Binomial Distribution

A situation is binomial when:

- there is a fixed number of trials
- each trial has two possible outcomes
- trials are independent
- probability of success is constant

For example:

> In 10 login attempts, what is the probability that exactly 3 are malicious?

This can be represented using a binomial distribution if the required assumptions are satisfied.

#### ⭐ Special Note

Remember:

\[
\boxed{\text{Binomial} = \text{number of successes in } n \text{ trials}}
\]

---

### Effect of \(n\) and \(p\) on Binomial Shape

The appearance of the binomial distribution depends on:

- \(n\), number of trials
- \(p\), probability of success

When \(p\) is around:

\[
0.5
\]

the distribution may appear relatively symmetric.

When \(p\) is close to 0 or 1, the distribution becomes more skewed.

Increasing the number of trials generally changes the distribution shape and can make it look smoother.

---

### Poisson Distribution

The **Poisson Distribution** models how many times an event occurs within a fixed interval of:

- time
- distance
- area
- space

Examples include:

- number of attacks per hour
- number of network failures per day
- number of incoming requests per second
- number of errors during a fixed time period

The probability is:

\[
\boxed{
P(X=k)=
\frac{e^{-\lambda}\lambda^k}{k!}
}
\]

where:

- \(k\) = number of events
- \(\lambda\) = average event rate

---

### Meaning of Lambda \(\lambda\)

In the Poisson distribution:

\[
\boxed{\lambda=\text{average number of events per interval}}
\]

For example:

If a server receives an average of:

\[
5
\]

attack attempts per hour, then:

\[
\lambda=5
\]

The Poisson distribution can then estimate probabilities such as:

> What is the probability of exactly 3 attacks in one hour?

---

### Shape of Poisson Distribution

The distribution shape changes depending on \(\lambda\).

For a small \(\lambda\), the distribution tends to be strongly right-skewed.

As \(\lambda\) increases, the distribution becomes wider and tends to appear more symmetric.

#### ⭐ Special Note

Poisson is especially useful when dealing with:

\[
\boxed{\text{counts of events occurring over a fixed interval}}
\]

---

### Binomial vs Poisson

These two can be confusing.

#### Binomial

Concerned with:

> How many successes occur in a fixed number of trials?

Example:

\[
10 \text{ login attempts}
\]

#### Poisson

Concerned with:

> How many events occur during a fixed interval?

Example:

\[
\text{attacks during one hour}
\]

A simple memory aid:

\[
\boxed{
\text{Binomial} \rightarrow \text{fixed trials}
}
\]

\[
\boxed{
\text{Poisson} \rightarrow \text{fixed interval}
}
\]

---

### Exponential Distribution

The final major distribution discussed was the **Exponential Distribution**.

The exponential distribution is closely related to the Poisson distribution.

Poisson answers:

> How many events occur in an interval?

Exponential answers:

> How long do we wait between events?

Its Probability Density Function is:

\[
\boxed{
f(x)=\lambda e^{-\lambda x}
}
\]

for:

\[
x\geq0
\]

where:

\[
\lambda = \text{event rate}
\]

---

### Example of Exponential Distribution

Suppose attacks arrive randomly.

The Poisson distribution may be used to model:

\[
\text{number of attacks per hour}
\]

The exponential distribution can model:

\[
\text{time between consecutive attacks}
\]

Other examples include:

- time between network requests
- waiting time between failures
- time between customers arriving
- time between security events

---

### Relationship Between Poisson and Exponential

This was an important conceptual relationship.

#### Poisson

Counts the number of events:

\[
\boxed{\text{How many events?}}
\]

#### Exponential

Models waiting time:

\[
\boxed{\text{How long until the next event?}}
\]

Both use an event rate:

\[
\lambda
\]

and are closely connected when events occur randomly and independently.

#### ⭐ Special Note

This distinction is worth remembering:

\[
\boxed{
\text{Poisson = Event Count}
}
\]

\[
\boxed{
\text{Exponential = Waiting Time}
}
\]

---

### Effect of Lambda on Exponential Distribution

For:

\[
f(x)=\lambda e^{-\lambda x}
\]

a higher value of \(\lambda\) means events occur more frequently.

Therefore, expected waiting times become shorter.

A lower \(\lambda\) means events occur less frequently, resulting in longer waiting times.

---

### Probability Distributions in Cybersecurity

The distributions discussed in this lecture can be useful in security-related analysis.

Examples:

**Normal Distribution**
- normal operating measurements
- latency
- processing times

**Uniform Distribution**
- cryptographic random values
- random key generation

**Binomial Distribution**
- successful attacks among a number of attempts
- malware detection success/failure

**Poisson Distribution**
- number of security events per hour
- number of attacks during a period

**Exponential Distribution**
- time between attacks
- time between network events

This shows why statistical distributions are useful for Machine Learning and cybersecurity analytics.

---

### ⭐ Special Notes / Lecturer Emphasis

For revision, focus especially on these points:

1. **Not all data follows a normal distribution.** Always inspect the distribution.
2. Understand skewness:

\[
\boxed{\text{Long right tail = Positive skew}}
\]

\[
\boxed{\text{Long left tail = Negative skew}}
\]

3. Histograms and KDE plots help visually examine the shape of data.
4. Normality can also be checked statistically using tests such as:
   - Shapiro-Wilk
   - Kolmogorov-Smirnov
5. For the Shapiro-Wilk test, a commonly used interpretation is:

\[
\boxed{p>0.05 \Rightarrow \text{do not reject normality}}
\]

6. **Uniform distribution:** every possible value in the interval has equal probability.
7. **Binomial distribution:** fixed number of independent trials with two possible outcomes.

\[
P(X=k)=
{n\choose k}p^k(1-p)^{n-k}
\]

8. **Poisson distribution:** models the number of events in a fixed interval.

\[
P(X=k)=
\frac{e^{-\lambda}\lambda^k}{k!}
\]

9. In Poisson distribution:

\[
\boxed{\lambda=\text{average event rate}}
\]

10. **Exponential distribution:** models the waiting time between events.

\[
f(x)=\lambda e^{-\lambda x}
\]

11. The important relationship is:

\[
\boxed{
\text{Poisson} \rightarrow \text{number of events}
}
\]

\[
\boxed{
\text{Exponential} \rightarrow \text{time between events}
}
\]

12. Know the difference between **discrete** and **continuous** distributions.

---

### Very Short Revision Summary

The main distributions in this lecture can be remembered like this:

| Distribution | Main Question |
|---|---|
| **Normal** | How are continuous values distributed around an average? |
| **Uniform** | Are all values equally likely? |
| **Binomial** | How many successes occur in \(n\) trials? |
| **Poisson** | How many events occur in an interval? |
| **Exponential** | How long until the next event? |

Key formulas:

\[
\boxed{
\text{Binomial: }
P(X=k)={n\choose k}p^k(1-p)^{n-k}
}
\]

\[
\boxed{
\text{Poisson: }
P(X=k)=\frac{e^{-\lambda}\lambda^k}{k!}
}
\]

\[
\boxed{
\text{Exponential: }
f(x)=\lambda e^{-\lambda x}
}
\]

The **core lesson of Lecture 3 Part 2** is that understanding the underlying probability distribution of data helps us choose appropriate preprocessing, statistical analysis, and Machine Learning techniques.

