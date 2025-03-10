Machine Learning project
==============

## Group: ML_Capybara

## Table of Contents
- [Introduction](#1-introduction)
- [Group memebers](#2-group-members-and-tasks-distribution)
- [About dataset](#3-about-dataset)
- [Process pipeline](#4-process-pipeline)
- [Model deployment](#5-model-deployment)
- [Model evaluation](#6-model-evaluation)

## 1. Introduction

In the assignment of Machine Learning course, we are required to apply models from chapter 2-6, defined and solved problem on a single dataset.

These 5 models include:

- Decision Tree (Chapter 2)

- Artifical Neural Network (Chapter 3)

- Naive Bayes (Chapter 4)

- Genetic Algorithm (Chapter 5)

- Graphical Models (Chapter 6)

## 2. Group members and tasks distribution

| Member name | ID | Task |
|----------|----------|----------|
| Tran Quoc Hieu | 2252217 | ... |
| Tran Quoc Trung | 2252859 | ... |
| Nguyen Anh Khoa | 2252352 | ... |
| Do Quang Hao | 2252352 | ... |
| Luu Chi Cuong | 2252352 | ... |

## 3. About dataset

- In this assignment, we use the dataset for `sentiment analysis` task.

- `sentiment analysis` is the task to classify sentiment (positive, neural, negative) given a document (a list of sentence).

- Our dataset includes more than 3500 examples with about 9 columns. However, for sentiment classification task, we only use 2 columns `texts` (data) and `sentiment` (label). For instance, the summary of this dataset is as follows:

|   | Column            | Non-Null Count | Dtype   |
|----|-------------------|---------------|-------- |
| 0  | textID           | 3534 non-null  | object  |
| 1  | text             | 3534 non-null  | object  |
| 2  | sentiment        | 3534 non-null  | object  |
| 3  | Time of Tweet    | 3534 non-null  | object  |
| 4  | Age of User      | 3534 non-null  | object  |
| 5  | Country          | 3534 non-null  | object  |
| 6  | Population -2020 | 3534 non-null  | float64 |
| 7  | Land Area (Km²)  | 3534 non-null  | float64 |
| 8  | Density (P/Km²)  | 3534 non-null  | float64 |

## 4. Process pipeline

For each model, we follow the below pipeline:

1. *Data preprocessing*: we utilizes natural language processing (NLP) techniques such as text preprocessing and TF-IDF vectorization for feature extraction.

2. *Model tuning*: we utilizes Optuna, a hyperparameter optimization framework. Model is tuned in training set using cross-validation technique. An abstract class of hyperparameter tuning is setup in a separate directory.

3. *Model training*: after fining optimizing hyperparameter, the model is then be fitted in training set.

4. *Model evaluation*: we use the following evaluation metrics:

    + Accuracy: Measures the percentage of correctly predicted labels out of all predictions.

    + F1-Score: A weighted average of precision and recall, useful for imbalanced datasets.
        
    + AUC-ROC: The Area Under the Receiver Operating Characteristic Curve, which evaluates the model's ability to distinguish between classes.


## 5. Model deployment

- For ease of observation, each model is implemented in jupyter notebook in separated directories inside *src/models*, the path is as follows:

    + Decision Tree ([here](src/models/decision_tree/README.md))

    + Artificial Neural Network ([here](src/models/MLP%20model/README.md))

    + Naives Bayes ([here](src/models/naive_bayes_model/README.md))

    + Genetic Algorithm ([here](src/models/genetic_algorithm/README.md))

    + Graphical Models ([here](src/models/BN%20model/))

## 6. Model evaluation

<table>
    <thead>
        <tr>
            <th rowspan=2>Model</th>
            <th rowspan=2>Accuracy</th>
            <th rowspan=2>F1-score</th>
            <th colspan=3>ROC-AUC</th>
        </tr>
        <tr>
            <th>Negative</th>
            <th>Neural</th>
            <th>Positive</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>Decision Tree</td>
            <td>0.6243</td>
            <td>0.6255</td>
            <td>0.7044</td>
            <td>0.6165</td>
            <td>0.7459</td>
        </tr>
        <tr>
            <td>MLP</td>
            <td>0.4040</td>
            <td>0.1918</td>
            <td>0.8060</td>
            <td>0.6742</td>
            <td>0.7613</td>
        </tr>
        <tr>
            <td>Naive Bayes</td>
            <td>0.4305</td>
            <td>0.3868</td>
            <td>0.6271</td>
            <td>0.5424</td>
            <td>0.6361</td>
        </tr>
        <tr>
            <td>Genetic Algorithm</td>
            <td>0.3126</td>
            <td>0.1665</td>
            <td>0.5044</td>
            <td>0.4918</td>
            <td>0.5233</td>
        </tr>
        <tr>
            <td>Bayes Network</td>
            <td>0.55</td>
            <td>0.5537</td>
            <td>0.80</td>
            <td>0.78</td>
            <td>0.64</td>
        </tr>
    </tbody>
</table>

