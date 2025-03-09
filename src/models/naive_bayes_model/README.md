# Text Classification Model using Gaussian Naive Bayes 

## 1. Overview

Naive Bayes is a machine learning model based on Bayesian theory. It assumes that every attributes are independent given the class (label). Likewise, Gaussian Naive Bayes (GNB) is a version of Naive Bayes to deal with continuous features. In this project, we setup a GNB model for sentiment classification task. We utilize natural language processing (NLP) techniques such as text preprocessing and TF-IDF vectorization for feature extraction.

## 2. Key features

Since GNB is already supported by `sklearn`, we only do preprocessing step, feature extraction and fit them into model. The feature includes the TF-IDF vectors, extracted from the `text` column of dataset.

## 3. Hyperparameter tuning

In this model, we consider the following hyperparameter for tuning:

- **MAX_FEATURES**: maximum number of word counts extracted from the text

- **NB_var_smoothing**: the smoothing variable of Naive Bayes model

## 4. Results 

- **Accuracy**: 0.4305

- **F1-score**: 0.3868

- **AUC-ROC**: 
    + Negative: 0.6271
    + Neural: 0.5424
    + Positive: 0.6361

### Result analysis

The model correctly classifies about 44.19% correct. This is not a high performance but is not terrible, compared with base model (predict all element as 'neural' class - 40.40%).

### Strengthness

- **Fast & Efficient**: GNB is extremely fast, even on large dataset.

- **Works Well with Small Datasets**: Compared with neural network model, GNB has smaller number of parameters, hence requires less data for training.

- **Handles High-Dimensional Data Well**: Since it treats features independently, GNB can handle high-dimensional spaces effectively (e.g., text classification with thousands of features).

### Weakness

- **Strong Independence Assumption**: GNB assumes that all features are independent, which is rarely true in real-world data. Hence, it may not capture interactions among features (data's correlation).

- **Poor Performance on Complex Decision Boundaries**: GNB performs poorly when class boundaries are complex or nonlinear, as it always assumes a simple decision surface.

- **Sensitive to Incorrect Distribution Assumptions**: GNB, or every Naive Bayes in general, only assumes for a certain distribution. Hence, if the real distribution is incorrect, the model will gain poor result.


