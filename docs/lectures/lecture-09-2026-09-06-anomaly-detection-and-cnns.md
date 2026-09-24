# Machine Learning — Lecture Notes

**Lecture date:** September 6, 2026  
**Main topics:** Anomaly Detection, Gaussian Models, Isolation Forests, Deep Learning, and Convolutional Neural Networks (CNNs)

## 1. Lecture Overview

This lecture had two main parts:

1. **Anomaly detection** using Gaussian probability models and Isolation Forests.
2. **Deep learning**, with a focus on how Convolutional Neural Networks automatically learn features from images and other signals.

The practical work used Python notebooks, Google Colab, TensorFlow/Keras, and examples from network security, image classification, and malware detection.

> ⭐ **Special Note:** Anomaly detection is useful when abnormal cases are rare or take many different forms. CNNs are useful when important features are difficult to design manually.

---

## 2. Anomaly Detection

Anomaly detection identifies observations that are unusual compared with normal data. Examples include:

- fraudulent financial transactions;
- unusual network traffic;
- abnormal heat or vibration in manufactured equipment;
- unusual CPU, memory, or network activity; and
- early warning signs for predictive maintenance.

It sits between supervised and unsupervised learning:

- The model can learn the normal pattern without using class labels.
- Labeled anomalies are still useful for validation, threshold selection, and final evaluation.

> ⭐ **Special Note:** An anomaly is not automatically fraud or a failure. It is an observation that deserves further investigation.

### 2.1 Why anomaly detection can outperform classification

Classification learns recurring labeled patterns. It may struggle when anomalies are extremely rare or when each abnormal case is different. Anomaly detection instead asks whether a new observation is sufficiently unlikely under the learned normal pattern. This means it can sometimes detect a previously unseen type of anomaly.

---

## 3. Gaussian Anomaly Detection

The Gaussian method assumes that each selected feature approximately follows a normal distribution. For a feature \(x_j\), estimate its mean and variance from the training data:

\[
\mu_j=\frac{1}{m}\sum_{i=1}^{m}x_j^{(i)}
\]

\[
\sigma_j^2=\frac{1}{m}\sum_{i=1}^{m}\left(x_j^{(i)}-\mu_j\right)^2
\]

The probability density of a new value is:

\[
p(x_j)=\frac{1}{\sqrt{2\pi}\sigma_j}
\exp\left(-\frac{(x_j-\mu_j)^2}{2\sigma_j^2}\right)
\]

For several features, the basic model multiplies their individual probabilities:

\[
p(x)=\prod_{j=1}^{n}p(x_j)
\]

The decision rule is:

\[
p(x)<\varepsilon \Rightarrow \text{anomaly}
\]

where \(\varepsilon\) is a chosen threshold.

### Basic workflow

1. Select features that may reveal abnormal behavior.
2. Estimate the mean and variance of each feature using normal training data.
3. Calculate the probability of each new observation.
4. Try several values of \(\varepsilon\) on a validation set.
5. Choose the threshold that gives the best evaluation result.
6. Use the test set once for the final unbiased evaluation.

> ⭐ **Special Note:** The threshold \(\varepsilon\) is a **hyperparameter**. It is selected by the practitioner rather than learned directly as a model parameter.

### 3.1 Feature engineering

Useful features may be raw measurements or combinations designed with domain knowledge. For example, the ratio of CPU activity to network traffic may reveal behavior that neither measurement shows alone.

If a feature is strongly skewed, a transformation such as a logarithm may make it closer to a Gaussian distribution. Polynomial or interaction features can also help separate normal and abnormal observations.

> ⭐ **Special Note:** Do not remove outliers automatically when the task is anomaly detection—the outliers may be the observations the model is supposed to find.

---

## 4. Multivariate Gaussian Model

The simple Gaussian method treats the features as independent. That assumption may be unrealistic; CPU usage and memory usage, for example, may be related.

A multivariate Gaussian model represents the feature relationships with a covariance matrix \(\Sigma\):

\[
p(x)=\frac{1}{(2\pi)^{n/2}|\Sigma|^{1/2}}
\exp\left(-\frac{1}{2}(x-\mu)^T\Sigma^{-1}(x-\mu)\right)
\]

- The diagonal of \(\Sigma\) contains the feature variances.
- The off-diagonal values describe how pairs of features vary together.
- Positive covariance means two features tend to increase together.
- Negative covariance means one tends to decrease as the other increases.

The same threshold rule is then applied to the resulting joint probability.

---

## 5. Isolation Forest

Isolation Forest detects anomalies by repeatedly making random splits in the feature space.

### Main intuition

- Normal observations are located in dense regions and usually require many splits to isolate.
- Anomalies are sparse and can often be isolated after only a few splits.
- Therefore, a **short average path length** across many trees indicates a likely anomaly.

The method builds several randomized isolation trees and combines them as a forest. Using several trees makes the result more stable than relying on one tree.

Important settings include:

- the number of trees;
- the sample size used to build each tree; and
- the anomaly-score or contamination threshold.

> ⭐ **Special Note:** Gaussian detection evaluates how improbable a sample is; Isolation Forest evaluates how easily it can be separated from the rest of the data.

---

## 6. Evaluation of Anomaly-Detection Models

Anomaly datasets are normally very imbalanced, so accuracy can be misleading. If only 1% of transactions are fraudulent, a model that always predicts “normal” still achieves 99% accuracy.

More useful measures include:

- **Precision:** of all observations predicted as anomalies, how many were truly anomalous?
- **Recall:** of all true anomalies, how many did the model detect?
- **F1-score:** the harmonic mean of precision and recall.
- **Confusion matrix:** shows true positives, true negatives, false positives, and false negatives.
- **Precision–recall curve:** shows the trade-off produced by changing the threshold.
- **Area under the precision–recall curve:** provides a single summary score; a value closer to 1 is better.

> ⭐ **Special Note:** Tune the threshold on the validation set. Do not repeatedly tune against the test set, because this causes the evaluation to become biased.

---

## 7. Machine Learning and Deep Learning

Traditional machine-learning solutions often rely on people to choose or engineer useful input features. Deep learning aims to learn both:

1. a useful representation of the raw data; and
2. the final prediction function.

This is particularly valuable for unstructured data such as images, audio, text, video, and raw signals, where manual feature extraction is difficult.

> ⭐ **Special Note:** The central advantage of deep learning is **automatic feature extraction**. Earlier layers learn simple patterns, while deeper layers combine them into higher-level representations.

The lecture connected the modern deep-learning resurgence to major advances in image classification and later to attention-based Transformer architectures used for language models.

---

## 8. Convolution and Image Features

An image can be represented numerically:

- a grayscale image has one channel;
- an RGB image has three channels; and
- every pixel contains an intensity or color value.

A convolution moves a small matrix called a **kernel** or **filter** across the input. At each location, it multiplies corresponding values and sums them to produce an output value.

Traditional image processing uses fixed filters to find patterns such as horizontal and vertical edges. A CNN makes the filter values learnable, allowing the model to discover the features most useful for the task.

> ⭐ **Special Note:** In a CNN, the kernel values are model parameters learned during training—not manually fixed edge detectors.

---

## 9. CNN Architecture

A typical CNN contains the following stages:

### 9.1 Convolution layer

Applies several learnable filters to produce multiple feature maps. Different filters learn different patterns.

### 9.2 Activation function

Introduces non-linearity so the model can learn complex decision boundaries. A common choice is ReLU:

\[
\operatorname{ReLU}(z)=\max(0,z)
\]

### 9.3 Pooling layer

Reduces the spatial dimensions of the feature maps. Max pooling keeps the largest value from each local region, reducing computation while retaining strong features.

### 9.4 Repeated feature extraction

Repeated convolution and pooling layers form a hierarchy:

- early layers learn edges and simple textures;
- middle layers combine them into shapes or parts; and
- deeper layers learn task-specific high-level features.

### 9.5 Flattening and dense layers

The final feature maps are flattened into a vector and passed to fully connected layers for classification.

### 9.6 Softmax output

For multiclass classification, softmax converts the output scores into probabilities:

\[
P(y=k)=\frac{e^{z_k}}{\sum_j e^{z_j}}
\]

The class with the highest probability is selected.

---

## 10. Training a CNN

Training begins with randomly initialized parameters. For each batch:

1. Images pass forward through the network.
2. Predictions are compared with the true labels.
3. A loss function measures the error.
4. Backpropagation calculates how each parameter contributed to the error.
5. An optimizer updates the parameters to reduce the loss.

For multiclass classification, the example used categorical cross-entropy. Training was performed in mini-batches, and one complete pass through the training data was called an **epoch**.

The practical example used:

- TensorFlow with the Keras interface;
- Google Colab with GPU acceleration;
- convolution, pooling, flattening, and dense layers;
- ReLU activations and softmax output;
- normalized pixel values; and
- training and validation curves to monitor performance.

> ⭐ **Special Note:** Apply exactly the same preprocessing to training, validation, test, and future input data. For image pixels, the example normalized values by dividing by 255.

---

## 11. Overfitting and Regularization

Overfitting occurs when the model learns the training data too closely and fails to generalize. A common warning sign is:

- training loss continues to decrease; but
- validation loss stops improving and begins to increase.

Methods discussed for controlling overfitting include:

- **dropout**, which temporarily disables some neurons during training;
- **early stopping**, which stops training when validation performance no longer improves;
- **regularization**, which adds a penalty term to the loss function;
- collecting more varied training examples; and
- selecting an appropriate number of epochs.

> ⭐ **Special Note:** More epochs do not always produce a better model. Stop near the point where validation performance is best.

After training, save the learned weights so that the model can make predictions without being retrained each time.

---

## 12. Applications Beyond Image Classification

Although CNNs became popular through computer vision, convolution can also be applied to:

- audio and ECG signals;
- malware binaries or API-call sequences;
- network-security data;
- medical imaging;
- object detection and image segmentation; and
- video, using three-dimensional convolution to include time.

Other neural-network families are better suited to different structures:

- recurrent networks and LSTMs for sequential data;
- Transformers for language and other sequences; and
- graph neural networks for relationships such as social or molecular graphs.

The shared idea is to transform complex unstructured input into a useful numeric representation learned from data.

---

## 13. Key Takeaways

- Anomaly detection finds rare or unusual observations rather than only repeating known labeled patterns.
- Gaussian models flag samples with probabilities below a chosen threshold.
- A multivariate Gaussian can model relationships between features through covariance.
- Isolation Forest identifies observations that can be separated with short tree paths.
- Accuracy alone is unreliable for heavily imbalanced data; use precision, recall, F1, and precision–recall curves.
- Deep learning learns features automatically from raw or minimally processed data.
- CNNs combine convolution, activation, pooling, flattening, and dense layers.
- Backpropagation and an optimizer learn the network parameters by minimizing loss.
- Validation curves, dropout, regularization, and early stopping help prevent overfitting.
- The same machine-learning principles—data splitting, preprocessing consistency, evaluation, and hyperparameter tuning—still apply in deep learning.
