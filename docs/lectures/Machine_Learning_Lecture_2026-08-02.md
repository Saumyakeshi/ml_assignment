# Machine Learning Lecture Notes

**Lecture date:** August 2, 2026  
**Main topic:** Feature Engineering, Feature Scaling, Categorical Encoding, and Dimensionality Reduction

## 1. Features and Labels

In Machine Learning:

\[
X = \text{Features/Inputs}
\]

\[
y = \text{Label/Output}
\]

Example: For house-price prediction, the house price is the label. Size, location, number of rooms, and age of the property are possible features.

Too few features may produce weak predictions. However, too many irrelevant features create noise and make learning more difficult.

> ⭐ **Special Note:** More features are useful only when they provide relevant information. Unnecessary features can reduce model performance.

## 2. Feature Engineering

Feature engineering transforms raw data into meaningful information that a model can understand.

\[
\boxed{\text{Raw Data} \rightarrow \text{Meaningful Features} \rightarrow \text{ML Model}}
\]

A machine cannot automatically understand the context of every value. Useful patterns must therefore be extracted and represented clearly.

### Cybersecurity examples

- **Login data:** Convert individual log entries into failed-login count, successful-login count, time between attempts, login outside office hours, and internal/external IP status.
- **Phishing detection:** Extract the number of links, suspicious links, attachments, urgent language, unusual sender domains, and the sender's previous behaviour.
- **Network intrusion:** Use packets per minute, SYN-packet count, average packet size, destination port, and whether an IP is internal or external.
- **Malware detection:** Extract file modifications per minute, changed file extensions, registry changes, unusual PowerShell or command-line activity, host-file changes, and command-and-control connections.

> ⭐ **Special Note:** Do not give millions of raw logs directly to a model. Extract the important patterns and give those patterns as features.

> ⭐ **Special Note:** Good feature engineering can allow a simpler model to perform well while requiring less processing power.

## 3. Main Types of Features

### Raw Features

Values already available directly in the dataset.

Examples:

- username
- IP address
- timestamp
- department
- destination port
- login status

### Derived Features

New features calculated from raw values.

Examples:

- failed logins during the last ten minutes
- files modified per minute
- percentage of suspicious links
- packets sent per second
- number of changed file extensions

### Interaction Features

Created by combining two or more features. Individual values may appear harmless, while their combination indicates risk.

Examples:

- **Location + time:** Logins from London and Australia within five minutes indicate impossible travel.
- **Department + downloads:** A finance employee downloading thousands of files may be suspicious.
- **Destination port + packet volume:** A large volume of traffic to SMB port 445 may indicate abnormal activity.

### Temporal Features

Features related to time and changes over time.

Examples:

- login outside business hours
- number of emails received within ten minutes
- sudden increase in registry modifications
- sharp rise in file changes
- repeated login attempts within a short period

> ⭐ **Special Note:** A value may not be suspicious by itself. Its relationship with another feature, or how it changes over time, may reveal the real pattern.

## 4. Handling Missing Values

A model should not normally be trained with unexplained missing values. Common approaches include:

- **Mean imputation:** Replace a missing numerical value with the arithmetic mean.
- **Median imputation:** Replace it with the middle value of the ordered data.
- **Mode imputation:** Use the most frequently occurring category.
- **Forward fill:** Use the previous available value.
- **Backward fill:** Use the next available value.
- **Missing-value indicator:** Add a separate feature showing whether the original value was missing.
- **Remove rows or columns:** Appropriate only when little useful information will be lost.

> ⭐ **Special Note:** Do not remove every row containing a missing value without checking the consequences. This may eliminate most of the dataset.

## 5. Feature Scaling

Features can use very different numerical ranges. For example:

- failed-login count: 0–10
- CPU usage: 0–100
- network bytes: possibly tens of thousands

Without scaling, a model may give excessive importance to the feature containing the largest numbers.

### Min–Max Normalization

Min–max normalization usually converts values to the range \(0\) to \(1\):

\[
x' = \frac{x-x_{\min}}{x_{\max}-x_{\min}}
\]

It preserves the relative position of each value within the original range.

### Z-Score Standardization

\[
z = \frac{x-\mu}{\sigma}
\]

Where:

- \(x\) is the original value
- \(\mu\) is the mean
- \(\sigma\) is the standard deviation

The standard deviation measures how widely the values are distributed around the mean.

\[
\sigma = \sqrt{\frac{\sum (x_i-\mu)^2}{N}}
\]

> ⭐ **Special Note:** Large numerical values are not automatically more important. Scaling allows the model to compare features more fairly.

> ⭐ **Technical Clarification:** Normalization and standardization scale existing numerical values; they do not directly fill missing values. Missing-value handling should normally happen before scaling.

## 6. Outliers

An outlier is a value that is unusually far from the rest of the data.

Outliers may represent:

- an error
- unusual but legitimate behaviour
- an attack or anomaly
- a rare event important to the prediction

> ⭐ **Special Note:** Do not automatically delete every outlier. In cybersecurity, the outlier may be the attack the model is supposed to detect.

## 7. Encoding Categorical Variables

Machine Learning models generally require numerical input. Categories must therefore be encoded.

### Ordinal Encoding

Use numerical values when the categories have a meaningful order.

Example:

\[
\text{High School}=0,\quad
\text{Bachelor's}=1,\quad
\text{Master's}=2
\]

### One-Hot Encoding

Use a separate binary column for each category when there is no natural order.

| Employee | Finance | IT | HR |
|---|---:|---:|---:|
| John | 1 | 0 | 0 |
| Bob | 0 | 1 | 0 |
| Alice | 0 | 0 | 1 |

> ⭐ **Special Note:** Do not encode unordered categories as Finance = 1, IT = 2, and HR = 3. The model may incorrectly conclude that one department is greater than another. Use one-hot encoding instead.

## 8. Dimensionality Reduction

Dimensionality reduction decreases the number of features while retaining the most useful information.

Benefits include:

- faster model training
- reduced processing requirements
- less noise and redundancy
- reduced overfitting
- easier visualization
- easier identification of important patterns
- reduced effect of the **curse of dimensionality**

The curse of dimensionality occurs when too many features make the dataset sparse and make useful patterns harder to learn.

### Feature Selection vs Feature Extraction

- **Feature selection:** Keep only the most useful original features.
- **Feature extraction:** Combine or transform original features into a smaller set of new features.

### Principal Component Analysis — PCA

PCA transforms correlated features into new components that retain as much variation as possible. It effectively changes or rotates the feature axes to find directions that explain the most information.

> ⭐ **Special Note:** Dimensionality reduction must preserve meaningful information. Reducing the number of columns is not useful if important patterns are removed.

## 9. Correlation and Multicollinearity

Highly correlated features provide similar information.

Examples:

- age and years of experience
- monthly salary and annual salary
- weight in kilograms and weight in grams
- URL count, hyperlink count, and total link count

Including several versions of the same information can cause:

- multicollinearity
- redundancy
- overfitting
- misleading feature importance
- confusion about which feature influences the prediction

Possible solutions include:

- removing one correlated feature
- combining related features
- feature selection
- PCA

> ⭐ **Special Note:** Duplicate or strongly related features do not necessarily improve a model. They can increase complexity without adding new information.

## 10. Chi-Square Test and ANOVA

The lecture briefly introduced statistical methods for selecting useful features.

### Chi-Square Test

The Chi-square test examines whether two categorical variables are associated.

Example:

- age group
- preferred social-media platform

The test compares observed values against expected values and uses a **p-value** to determine whether the relationship is statistically significant.

### ANOVA

ANOVA can be used when comparing a numerical value across multiple groups.

The lecturer indicated that hypothesis testing, p-values, Chi-square, and ANOVA would be examined in more detail in the following class.

## 11. Assignment Guidance

The assignment involves creating Machine Learning solutions for four cybersecurity-related problems, including areas such as:

- email/phishing detection
- general cyberattack detection
- anomaly detection
- malware or ransomware detection

For each problem:

1. Select a suitable dataset.
2. Clean and feature-engineer the data.
3. Apply one classical Machine Learning algorithm.
4. Apply one Deep Learning algorithm.
5. Compare their performance.

Suggested evaluation measures include:

- accuracy
- precision
- recall
- F1-score
- AUC
- confusion matrix

> ⭐ **Assignment Note:** Use realistic datasets with enough records, features, and labels. A very small or overly simple dataset may produce unrealistically high results that do not demonstrate meaningful learning.

## Key Takeaways

\[
\boxed{\text{Feature engineering converts raw data into learnable patterns}}
\]

- Features are inputs; labels are outputs.
- Raw data needs context before a model can use it effectively.
- Derived, interaction, and temporal features often reveal more than individual raw values.
- Handle missing values before training.
- Scale numerical features so that large values do not dominate.
- Use one-hot encoding for unordered categories.
- Remove or combine redundant and highly correlated features.
- Reduce dimensions carefully while preserving important information.
- Evaluate cybersecurity models using several metrics, not accuracy alone.

> ⭐ **Most Important Lecture Message:** The quality of a Machine Learning model depends heavily on the quality of the features supplied to it. A sophisticated algorithm cannot compensate for poorly prepared, noisy, or meaningless input data.
