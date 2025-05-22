# Text Classification Model using Logistic Regression

## 1. Overview

Logistic Regression is a basic machine learning model used in the classification task. For a given sample, it does not directly predict the target class but rather the probability of each class which is a real value between 0 and 1. It follows the principle of *Maximum Entropy* - maximizing likelihood of given data.

## 2. Key Features

Since our task is 3-classes classification, we implement multi-class logistic regression in which it uses **softmax** function instead of **sigmoid** (used for binary classification). For instance, our model predicts probabilities for each class. It maps outputs of the linear discrimant function, $y = Wx + b$, into probabilities (value between 0 and 1) that sum to 1 by passing $y$ through the softmax function. Given N samples {$(x_i,y_i) | i=\overline{1,N}$}, where $y_i \in $ {Positive, Neural, Negative}, the objective is to maximize
$$H = \prod_{i=1}^N P(y = y_i | x_i)$$

It is equivalent with minimizing the loss function:

$$L = - \sum_{i=1}^N y_i * \log(\hat{y_i})$$

## 3. Model Architecture

In our project, we implement the logistic regression model using the neural network block from `torch` library of python. For optimization, we use **Gradient Descent** to update parameters and **Early Stopping** in which the training process is break when the accuracy in the validation set does not improve after some epochs. 

The building block of this model is:

* 1 Fully connected layer with shape $input \times output$

* Activation function: *ReLU*

* Softmax for generating probabilities-like

The output is always 3 probabilities for 3 classes. About the input features, in this section we implement 2 approaches:

1. Tf-idf of 50,000 most frequently grams (view *logistic_regression.ipynb* notebook of the same direction).

2. Word-vectorizing with dimension reduction using PCA from [PCA section](../dimension_reduction/README.md) (view *logistic_regression.ipynb* notebook of the same direction). 

For the 2nd approach, unlike other hyperparameter in the next part, the number of features to be kept is more complex to be automatically tunned since it affects the early preprocessing of dataset. Hence, we choose to tune it manually in which we try many values to see which one is the best. The final number we kept is 20 features with about 60% remaining variance.

## 4. Hyperparameter tuning

* batch_size = 32

* num_epochs = 100

In these models, we setup hyperparameter tunning for 2 params:  *learning_rate* and *early_stopping_iter* where:

* *learning_rate*: defines how much gradient model update per each backward

* *early_stopping_iter*: defines the number of epochs that early stopping will be applied if model does not improve the validation accuracy.

## 5. Results

We shown the results of model prediction on test set of two models as follows:

<table>
    <thead>
        <tr>
            <th rowspan=2>Model</th>
            <th rowspan=2>Accuracy</th>
            <th rowspan=2>F1-score</th>
            <th colspan=3>ROC-AUC</th>
            <th rowspan=2>Training time per epoch (second)</th>
            <th rowspan=2>Learning rate</th>
            <th rowspan=2>Early stopping</th>
            <th rowspan=2>Number of epochs to converge</th>
        </tr>
        <tr>
            <th>Negative</th>
            <th>Neural</th>
            <th>Positive</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>Logistic Regression with Tf-idf</td>
            <td>0.6811</td>
            <td>0.6822</td>
            <td>0.8638</td>
            <td>0.7695</td>
            <td>0.8955</td>
            <td>12</td>
            <td>3e-3</td>
            <td>7</td>
            <td>14</td>
        </tr>
        <tr>
            <td>Logistic Regression with word vectorizing and PCA</td>
            <td>0.6347</td>
            <td>0.6315</td>
            <td>0.8130</td>
            <td>0.7378</td>
            <td>0.8149</td>
            <td>2</td>
            <td>0.041</td>
            <td>7</td>
            <td>20</td>
        </tr>
    </tbody>
</table>

### Comment on result

Applying PCA to reduce the dimensionality shows a clearly trade-off: the training time improves but the performace decreases.


## 6. Model strength

* Logistic regression is simple and fast. Since it directly connects input nodes to outputs, the data forwarding through the model is extremely fast, and hence the training time is small even the input feature is large (tf-idf with 50,000 features).

* Logistic regression is interpretable since it outputs are probabilities. Also, by examining weights, we can know which features significantly contribute to the result.

* It prevents the overfitting due to the model complexity.


## 7. Model weakness

* Since logistic regression is a simplicity model, it can not capture the whole underlying data distribution which may leads to underfitting. The result accuracy which is only around 0.6811 shows that weakness.
