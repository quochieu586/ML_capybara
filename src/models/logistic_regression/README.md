# Text Classification Model using Logistic Regression

## 1. Overview

Logistic Regression is a basic machine learning model used in the classification task. For a given sample, it does not directly predict the target class but rather the probability of each class which is a real value between 0 and 1. It follows the principle of *Maximum Entropy* - maximizing likelihood of given data.

## 2. Key Features

Since our task is 3-classes classification, we implement multi-class logistic regression in which it uses **softmax** function instead of **sigmoid** (used for binary classification). For instance, our model predicts probabilities for each class. It maps outputs of the linear discrimant function, $y = Wx + b$, into probabilities (value between 0 and 1) that sum to 1 by passing $y$ through the softmax function. Given N samples $\left\{(x_i,y_i) | i=\overline{1,N} \right\}$, where $y_i \in $ {Positive, Neural, Negative}, the objective is to maximize
$$H = \prod_{i=1}^N P(y = y_i | x_i)$$

It is equivalent with minimizing the loss function:

$$L = - \sum_{i=1}^N y_i * \log(\hat{y_i})$$

## 3. Model Architecture

In our project, we implement the logistic regression model using the neural network block from `torch` library of python. For optimization, we use **Gradient Descent** to update parameters. The model architecture is as follows:

* Input feature: Tf-idf of 50,000 most frequently grams

* Activation function: *ReLU*

* Probabilities generation: *Softmax*

* Output: 3 probabilites for 3 classes, respectively.



## 4. Hyperparameter

* learning_rate = 1e-3

* batch_size = 32

* num_epochs = 20

* stop_iter = 3

## 5. Results

We shown the results of model prediction on test set as follows:

+ **Accuracy**: 0.680

+ **F1-score**: 0.681

+ **AUC-ROC**:
    + Negative: 0.8634
    + Neural: 0.7689
    + Positive: 0.8953

<!-- ### Comment on result -->


## 6. Model strength

* Logistic regression is simple and fast. Since it directly connects input nodes to outputs, the data forwarding through the model is extremely fast, and hence the training time is small even the input feature is large (tf-idf with 50,000 features).

* Logistic regression is interpretable since it outputs are probabilities. Also, by examining weights, we can know which features significantly contribute to the result.

* It prevents the overfitting due to the model complexity.


## 7. Model weakness

* Since logistic regression is a simplicity model, it can not capture the whole underlying data distribution which may leads to underfitting. The result accuracy which is only around 0.680 shows that weakness.
