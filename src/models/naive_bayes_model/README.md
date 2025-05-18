# Text Classification Model using Gaussian Naive Bayes 

## 1. Overview

Naive Bayes is a machine learning model based on Bayesian theory. It assumes that every attributes are independent given the class (label). Likewise, Gaussian Naive Bayes (GNB) is a version of Naive Bayes to deal with continuous features. In this project, we setup a GNB model for sentiment classification task. We utilize natural language processing (NLP) techniques such as text preprocessing and TF-IDF vectorization for feature extraction.

## 2. Key features

Since GNB is already supported by `sklearn`, we only do preprocessing step, feature extraction and fit them into model.For the input feature, we setup 2 approaches:

* TF-IDF vectors, extracted from the `text` column of dataset.

* Word embedding.

For each of them, we compare two versions: full input and reduced input using PCA (from section [dimension reduction](../dimension_reduction/README.md)).

## 3. Hyperparameter tuning

In this model, we consider the following hyperparameter for tuning:

- **MAX_FEATURES**: maximum number of word counts extracted from the text

- **NB_var_smoothing**: the smoothing variable of Naive Bayes model

## 4. Results

<table>
    <thead>
        <tr>
            <th rowspan=2>Model</th>
            <th rowspan=2>Accuracy</th>
            <th rowspan=2>F1-score</th>
            <th colspan=3>ROC-AUC</th>
            <th rowspan=2>Input dim</th>
        </tr>
        <tr>
            <th>Negative</th>
            <th>Neural</th>
            <th>Positive</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>NB using Tf-idf</td>
            <td>0.5068</td>
            <td>0.5224</td>
            <td>0.7252</td>
            <td>0.6844</td>
            <td>0.8002</td>
            <td>1000</td>
        </tr>
        <tr>
            <td>NB using Tf-idf with PCA</td>
            <td>0.4052</td>
            <td>0.4052</td>
            <td>0.6068</td>
            <td>0.5418</td>
            <td>0.6212</td>
            <td>485</td>
        </tr>
        <tr>
            <td>NB using word vectorizing</td>
            <td>0.5351</td>
            <td>0.5332</td>
            <td>0.7427</td>
            <td>0.6572</td>
            <td>0.7719</td>
            <td>300</td>
        </tr>
        <tr>
            <td>NB using word vectorizing with PCA</td>
            <td>0.5976</td>
            <td>0.5969</td>
            <td>0.8041</td>
            <td>0.7055</td>
            <td>0.8067</td>
            <td>20</td>
        </tr>
    </tbody>
</table>

### Result analysis

When using tf-idf as input feature, the model that uses reduced input performs lower performance as its parameter is 485 compared to 1000. However, for word embedding approach, it is extremely suprise that, model that use reduces input performs better than the original one. This result shows that 20 most important features is better for training than the whole 300.

### Strengthness

- **Fast & Efficient**: GNB is extremely fast, even on large dataset.

- **Works Well with Small Datasets**: Compared with neural network model, GNB has smaller number of parameters, hence requires less data for training.

- **Handles High-Dimensional Data Well**: Since it treats features independently, GNB can handle high-dimensional spaces effectively (e.g., text classification with thousands of features).

### Weakness

- **Strong Independence Assumption**: GNB assumes that all features are independent, which is rarely true in real-world data. Hence, it may not capture interactions among features (data's correlation).

- **Poor Performance on Complex Decision Boundaries**: GNB performs poorly when class boundaries are complex or nonlinear, as it always assumes a simple decision surface.

- **Sensitive to Incorrect Distribution Assumptions**: GNB, or every Naive Bayes in general, only assumes for a certain distribution. Hence, if the real distribution is incorrect, the model will gain poor result.


