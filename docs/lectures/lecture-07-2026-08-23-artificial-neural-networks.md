# Machine Learning Lecture Notes

**Lecture date:** August 23, 2026  
**Main topic:** Artificial Neural Networks, Perceptrons, Multilayer Perceptrons, Activation Functions, Backpropagation, and Overfitting

## 1. Why Neural Networks?

Artificial Neural Networks (ANNs) are computational models inspired by interconnected biological neurons. They are designed to learn complex relationships from examples.

Neural networks are especially useful for problems such as:

- image and facial recognition;
- handwriting recognition;
- speech or sound recognition;
- complex attack detection;
- network-traffic pattern recognition;
- other nonlinear problems with many interacting features.

Traditional ML models can perform well on structured problems, but a neural network can learn richer intermediate representations through its hidden layers.

> ⭐ **Special Note:** Neural networks do not eliminate the need for good data, preprocessing, validation, or feature understanding. They can still overfit.

## 2. Structure of an Artificial Neural Network

A basic feed-forward neural network contains:

1. an **input layer** that receives the features;
2. one or more **hidden layers** that transform the information;
3. an **output layer** that produces the final prediction.

A network’s:

- **depth** is the number of successive layers;
- **width** is the number of neurons in a layer.

Increasing depth or width increases model capacity, but also increases computation and the risk of overfitting.

> ⭐ **Technical Clarification:** More neurons or layers do not automatically produce better accuracy. Network size must be selected and validated for the problem.

## 3. The Artificial Neuron

A neuron receives inputs, combines them using weights and a bias, and applies an activation function.

The weighted sum is:

\[
z=b+\sum_{i=1}^{p}w_ix_i
\]

The neuron’s output is:

\[
a=f(z)
\]

Where:

- \(x_i\) is an input feature;
- \(w_i\) is its weight;
- \(b\) is the bias;
- \(f\) is the activation function;
- \(a\) is the activation or neuron output.

The weights represent how strongly the inputs influence the neuron. The bias shifts the activation boundary, allowing the neuron to learn a decision rule that does not have to pass through the origin.

> ⭐ **Technical Clarification:** Weights do not normally need to add up to 100%, and bias is not merely used to make an output positive. Both are learned parameters.

## 4. Activation Functions

An activation function determines how the weighted input is transformed before being passed to the next layer.

### Step Function

A simple perceptron may use a threshold:

\[
f(z)=
\begin{cases}
1,&z\geq0\\
0,&z<0
\end{cases}
\]

This is suitable for simple binary, linearly separable problems.

### Sigmoid

\[
\sigma(z)=\frac{1}{1+e^{-z}}
\]

The sigmoid maps a value to the range \((0,1)\). It is commonly used for the output of binary classification networks.

### Hyperbolic Tangent

\[
\tanh(z)=\frac{e^z-e^{-z}}{e^z+e^{-z}}
\]

Tanh maps values to \((-1,1)\).

### ReLU

\[
ReLU(z)=\max(0,z)
\]

ReLU is widely used in hidden layers because it is simple and often trains efficiently.

> ⭐ **Special Note:** The activation function provides nonlinearity. Without nonlinear activations, stacking several layers still behaves like a single linear transformation.

## 5. Perceptron

A perceptron is a single-neuron binary classifier. It can learn a linear decision boundary.

The lecture used logic gates to demonstrate its capabilities.

### AND Gate

| \(x_1\) | \(x_2\) | Output |
|---:|---:|---:|
| 0 | 0 | 0 |
| 0 | 1 | 0 |
| 1 | 0 | 0 |
| 1 | 1 | 1 |

Cybersecurity analogy: allow login only when both the username and password are correct.

### OR Gate

| \(x_1\) | \(x_2\) | Output |
|---:|---:|---:|
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 1 |

Cybersecurity analogy: raise an alert when either the firewall or endpoint-security tool detects suspicious activity.

### NOT Gate

| \(x\) | Output |
|---:|---:|
| 0 | 1 |
| 1 | 0 |

AND, OR, and NOT can be represented using linear decision boundaries.

## 6. XOR Problem

The XOR gate returns 1 only when its inputs differ:

| \(x_1\) | \(x_2\) | Output |
|---:|---:|---:|
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

No single straight line can separate the two output classes. XOR is therefore **not linearly separable** and cannot be learned by a single perceptron.

> ⭐ **Special Note:** XOR is the classic demonstration of why hidden layers and nonlinear activation functions are needed.

## 7. Multilayer Perceptron

A Multilayer Perceptron (MLP) contains:

- an input layer;
- at least one hidden layer;
- an output layer;
- learned weights and biases;
- nonlinear activation functions.

Hidden neurons can learn intermediate features. Later layers combine these learned representations to solve nonlinear problems such as XOR.

One sufficiently wide hidden layer can theoretically approximate many continuous functions, although deeper networks may learn some problems more efficiently.

## 8. Forward Propagation

During a forward pass:

1. the input features enter the network;
2. each neuron calculates its weighted sum;
3. the activation function produces an output;
4. that output becomes input to the next layer;
5. the output layer produces the prediction.

Forward propagation alone does not train the network. The prediction must be compared with the correct answer to calculate a loss.

## 9. Loss and Backpropagation

The loss function measures the difference between the model’s prediction and the true target.

Backpropagation efficiently calculates how each parameter contributed to the loss. Using the chain rule, it sends gradient information backward from the output layer through the hidden layers.

The general update is:

\[
w_{new}=w_{old}-\eta\frac{\partial L}{\partial w}
\]

Where:

- \(L\) is the loss;
- \(\eta\) is the learning rate;
- \(\frac{\partial L}{\partial w}\) is the gradient.

This cycle is repeated:

\[
\boxed{\text{Forward pass}\rightarrow\text{Loss}\rightarrow\text{Backpropagation}\rightarrow\text{Update}}
\]

> ⭐ **Special Note:** Backpropagation is not simply “sending an inaccurate neuron back.” It calculates gradients for all relevant parameters so they can be updated to reduce the overall loss.

## 10. Learning Rate, Epochs, and Momentum

### Learning Rate

The learning rate controls the size of each parameter update.

- Too small: learning is slow.
- Too large: the optimiser may overshoot the minimum or become unstable.

### Epoch

One epoch is one complete pass through the training dataset.

Training for more epochs is useful only while validation performance improves.

### Momentum

Momentum uses part of the previous update to smooth and accelerate optimisation. It can help the optimiser move through shallow regions or small local minima.

A simplified form is:

\[
v_t=\beta v_{t-1}-\eta\nabla L(w_t)
\]

\[
w_{t+1}=w_t+v_t
\]

The momentum coefficient \(\beta\) is often near 0.9, though it is a tunable hyperparameter.

## 11. Local and Global Minima

The loss surface may contain several low points.

- A **local minimum** is lower than its nearby points but is not the best possible solution.
- A **global minimum** is the lowest point across the entire loss surface.

Optimisers may become stuck or slow down in local minima, plateaus, or saddle points. Momentum and modern adaptive optimisers can help.

## 12. Understanding Learning Curves

The lecture discussed several possible loss patterns:

- **Healthy learning:** training loss decreases and eventually stabilizes at a low value.
- **Plateau:** loss stops improving, possibly because of poor features, unsuitable architecture, or optimisation settings.
- **Unstable learning:** loss repeatedly rises and falls, suggesting a learning-rate, scaling, data, or optimisation problem.
- **Overfitting:** training loss continues decreasing while validation loss begins increasing.

> ⭐ **Special Note:** Validation loss is more important than training loss for deciding whether the model generalizes.

Possible responses to overfitting include:

- early stopping;
- dropout;
- regularization;
- reducing model size;
- obtaining more representative data;
- data augmentation;
- better train/validation/test separation.

## 13. Training Data Requirements

A useful training dataset should:

- represent the population where the model will be used;
- contain examples from every important class;
- contain realistic variation within each class;
- avoid leakage between training, validation, and testing data;
- contain reliable labels;
- be sufficiently large for the network’s complexity.

> ⭐ **Special Note:** A powerful network cannot compensate for unrepresentative or incorrectly labelled training data.

## 14. Character-Recognition Example

Handwritten digit recognition demonstrates why neural networks are useful. The same digit may vary in:

- shape;
- stroke width;
- position;
- orientation;
- pixel density;
- writing style.

Input pixels or extracted image features are processed by hidden neurons, which learn intermediate patterns before the output layer predicts one of the ten digit classes.

## 15. Practical Demonstration

The lecture used NumPy, Matplotlib, and scikit-learn to demonstrate:

- Perceptron learning for the AND gate;
- plotting the linear decision boundary;
- failure of a single perceptron on XOR;
- using `MLPClassifier` for a nonlinear XOR problem.

The AND-gate demonstration produced the expected outputs. The XOR MLP demonstration did not converge to the correct predictions during the lecture.

This practical failure does not invalidate the theory. With only four training observations, MLP results can be sensitive to architecture, initialization, activation, optimiser, learning rate, and stopping conditions.

> ⭐ **Special Note:** When a neural-network example fails, inspect convergence warnings, loss history, architecture, activation, scaling, initialization, random seed, and hyperparameters instead of assuming the concept is wrong.

## 16. Assignment Guidance

The lecturer indicated that the next class would begin directly with an assignment question. A later assignment practical would compare models such as CNN and Random Forest on a cybersecurity dataset.

The XOR MLP practical was also left for correction or continuation in the following class.

## Key Takeaways

- A neuron computes a weighted sum, adds a bias, and applies an activation function.
- A perceptron can solve only linearly separable classification problems.
- XOR cannot be solved by a single perceptron.
- Hidden layers and nonlinear activation functions allow an MLP to learn nonlinear relationships.
- A forward pass produces predictions; backpropagation calculates gradients.
- Gradient-based optimisation updates weights to reduce loss.
- Training and validation curves must both be monitored.
- Neural networks can still overfit and require representative data and careful validation.

> ⭐ **Most Important Lecture Message:** Multilayer neural networks gain their power from learned hidden representations and nonlinear activations, but successful training still depends on suitable data, loss functions, optimisation, and validation.
