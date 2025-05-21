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

### Assigment 1

| Member name | ID | Task |
|----------|----------|----------|
| Tran Quoc Hieu | 2252217 | Genetic Algorithm |
| Tran Quoc Trung | 2252859 | Naives Bayes |
| Nguyen Anh Khoa | 2252352 | Graphical Models (Manually Naives Bayes) |
| Do Quang Hao | 2252180 | Artificial Neural Network |
| Luu Chi Cuong | 2252097 | Decision Tree |

### Assignment 2


## 3. About dataset

- In this assignment, we use the dataset for `sentiment analysis` task.

- `sentiment analysis` is the task to classify sentiment (positive, neural, negative) given a document (a list of sentence).

- This dataset includes training data with 27480 examples and test data with 3534 examples. Each has about 9 columns. For instance, the summary of this dataset is as follows:

|   | Column            | Non-Null Count | Dtype   |
|----|-------------------|---------------|-------- |
| 0  | textID           | 27480 non-null  | object  |
| 1  | text             | 27480 non-null  | object  |
| 2  | selected_text    | 27480 non-null  | object  |
| 3  | sentiment        | 27480 non-null  | object  |
| 4  | Time of Tweet    | 27480 non-null  | object  |
| 5  | Age of User      | 27480 non-null  | object  |
| 6  | Country          | 27480 non-null  | object  |
| 7  | Population -2020 | 27480 non-null  | float64 |
| 8  | Land Area (Km²)  | 27480 non-null  | float64 |
| 9  | Density (P/Km²)  | 27480 non-null  | float64 |

- For sentiment classification task, we only focus on 2 columns `texts` (data) and `sentiment` (label). The distribution of each classes in dataset is as follows:

| Class | Training | Test |
| ----- | ----- |
| neutral  | 11117 | 1430 |
| positive | 8582 | 1103 |
| negative | 7781 | 1001 | 


## 4. Process pipeline

For each model, we follow the below pipeline:

1. *Data preprocessing*: This step includes cleaning the text and extracting features from them. In our assignment, we implement 2 kinds of features: Tf-idf and Word Embedding. We shown later characteristics of each feature extration way.

2. *Model tuning*: we utilizes Optuna, a hyperparameter optimization framework. Model is tuned in training set using cross-validation technique. An abstract class of hyperparameter tuning is setup in a separate directory.

3. *Model training*: after fining optimizing hyperparameter, the model is then be fitted in training set.

4. *Model evaluation*: we use the following evaluation metrics:

    + Accuracy: Measures the percentage of correctly predicted labels out of all predictions.

    + F1-Score: A weighted average of precision and recall, useful for imbalanced datasets.
        
    + AUC-ROC: The Area Under the Receiver Operating Characteristic Curve, which evaluates the model's ability to distinguish between classes.


## 5. Model deployment

### Assignment 1

For ease of observation, each model is implemented in jupyter notebook in separated directories inside *src/models*, the path is as follows:

- Decision Tree ([here](src/models/decision_tree/))

- Artificial Neural Network ([here](src/models/MLP%20model/))

- Naives Bayes ([here](src/models/naive_bayes_model/))

- Genetic Algorithm ([here](src/models/genetic_algorithm/))

- Graphical Models ([here](src/models/BN%20model/))

### Assignment 2

For the second assignment, we continue to implement models from Chapter 6 - 10. Detail about each model implementation is put in the corresponding path as follows:

- Graphical Model - Hidden Markov model ([here](src/models/HMM/))

- Support Vector Machine ([here](src/models/SVM%20model/))

- Dimensional Reduction - Principal Components Analysis ([here](src/models/dimension_reduction/))

- Ensembler Method - Boosting ([here](src/models/EnsembleLearning/))

- Differential Model - Logistic Regression ([here](src/models/logistic_regression/))

## 6. Model evaluation

We show in this table the result on test set for each model. This includes the input feature (Tf-idf or word embedding) with its size, accuracy, f1-score, ROC-AUC (negative, neural, positive).

<table>
    <thead>
        <tr>
            <th rowspan=2>Model</th>
            <th colspan=2>Feature</th>
            <th rowspan=2>Accuracy</th>
            <th rowspan=2>F1-score</th>
            <th colspan=3>ROC-AUC</th>
        </tr>
        <tr>
            <th>Name</th>
            <th>Size</th>
            <th>Negative</th>
            <th>Neural</th>
            <th>Positive</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>Decision Tree (DT)</td>
            <td>Tf-idf</td>
            <td>25,828</td>
            <td>0.6919</td>
            <td>0.6915</td>
            <td>0.8297</td>
            <td>0.7743</td>
            <td>0.8682</td>
        </tr>
        <tr>
            <td>Multi-layer Perceptron (MLP)</td>
            <td>Tf-idf</td>
            <td>50,000</td>
            <td>0.6984</td>
            <td>0.6984</td>
            <td>0.8672</td>
            <td>0.7967</td>
            <td>0.8987</td>
        </tr>
        <tr>
            <td rowspan=4>Naive Bayes (NB)</td>
            <td rowspan=2>Tf-idf</td>
            <td>1,000</td>
            <td>0.5068</td>
            <td>0.5224</td>
            <td>0.7252</td>
            <td>0.6844</td>
            <td>0.8002</td>
        </tr>
        <tr>
            <!-- <td>Tf-idf</td> -->
            <td>485 (60% variance after applying PCA)</td>
            <td>0.4052</td>
            <td>0.4052</td>
            <td>0.6068</td>
            <td>0.5418</td>
            <td>0.6212</td>
        </tr>
        <tr>
            <td rowspan=2>Word embedding</td>
            <td>300</td>
            <td>0.5351</td>
            <td>0.5332</td>
            <td>0.7427</td>
            <td>0.6572</td>
            <td>0.7719</td>
        </tr>
        <tr>
            <!-- <td>Word embedding</td> -->
            <td>20 (60% variance after applying PCA)</td>
            <td>0.4052</td>
            <td>0.4052</td>
            <td>0.8041</td>
            <td>0.7055</td>
            <td>0.8067</td>
        </tr>
        <tr>
            <td>Genetic Algorithm (GA)</td>
            <td>Tf-idf</td>
            <td>1,000</td>
            <td>0.3126</td>
            <td>0.1665</td>
            <td>0.5044</td>
            <td>0.4918</td>
            <td>0.5233</td>
        </tr>
        <!-- <tr>
            <td>Bayes Network</td>
            <td>...</td>
            <td>...</td>
            <td>0.55</td>
            <td>0.5537</td>
            <td>0.80</td>
            <td>0.78</td>
            <td>0.64</td>
        </tr> -->
        <tr>
            <td>Hidden Markov Model (HMM)</td>
            <td>Tokens sequence</td>
            <td></td>
            <td>0.4434</td>
            <td>0.4372</td>
            <td>0.4836</td>
            <td>0.5388</td>
            <td>0.4763</td>
        </tr>
        <tr>
            <td>Support Vector Machine (SVM)</td>
            <td>Tf-idf</td>
            <td>2,878 (90% variance after applying PCA)</td>
            <td>0.6488</td>
            <td>0.6509</td>
            <td>0.8299</td>
            <td>0.7309</td>
            <td>0.8549</td>
        </tr>
        <tr>
            <td>Emsembler method (EM)</td>
            <td>Word embedding</td>
            <td>20 (60% variance after applying PCA)</td>
            <td>0.6178</td>
            <td>0.6203</td>
            <td>0.8044</td>
            <td>0.7404</td>
            <td>0.8350</td>
        </tr>
        <tr>
            <td rowspan=2>Logistic regression (LR)</td>
            <td>Tf-idf</td>
            <td>50,000</td>
            <td>0.6811</td>
            <td>0.6822</td>
            <td>0.8638</td>
            <td>0.7695</td>
            <td>0.8955</td>
        </tr>
        <tr>
            <td>Word embedding</td>
            <td>20 (60% variance after applying PCA)</td>
            <td>0.6347</td>
            <td>0.6315</td>
            <td>0.8130</td>
            <td>0.7378</td>
            <td>0.8149</td>
        </tr>
    </tbody>
</table>

In dimensional reduction section, we do not implement a specific model. Instead we introduce a reduced input feature which is used by other model and compared with original one.

## 7. Discussion 

- **Input feature**: In our project, we extract and use 2 different feature class: tf-idf and word-embedding. For each model, we choose to implement the suitable input feature approach. Large input feature is not valid for some models due to the hardware limitation. For example, with *EM* and *GA*, they include create a lot of models which cost a lot of space if we choose to implement large tf-idf feature. On the other hand, the input feature of HMM is different from these two due to its different behaviour. It works on the sequence of tokens and determine what the most likely sequence of states that generate this.

- **Hyperparameter tuning**: During implementing, we spend a part of data for model validation as well as hyperparameter tuning for best model. For example, we fine-tune the learning rate and number of hidden nodes for *MLP* and number of estimators as well as max growing depth for generating tree in *EM*. Tuning shows a non-trivial improvement for some model. On the other hand, for some others, due to hardware limitation, they can not be tuned properly leading to low result (*HMM, GA*).

- **Result comparision**: 

    + Among these models, not suprisingly, MLP with 1 hidden layer and 50,000 tf-idf input gives the best result since it has the largest number of parameters. However, as we introduce in the corresponding notebook, this model is not stable since it depends strongly on the initialization point and easy to overfit the dataset.

    + Logistic regression and Decision tree also show high performances compared with others. These are models that we can utilize the full tf-idf feature and tuning, hence give a better result.

    + On the other hand, *SVM* and *EM*, which use the reduced input, give an acceptable result. When implementing them, we observe that using the full input feature is time-costly and space-costly. Hence, reduced feature using PCA is applied in which it shows a reasonable result.

    + Finally, among them, *GA* and *HMM* are two models which have lowest performance. For *GA*, we abstract the string of bits by the string (flatten list) of parameters of neural network models. Since it requires creating many neural network models, it costs a lot of space in which we can only use a part (reduced) of input feature and also time-expensive since each training epoch required pass data through many neural networks. For *HMM*, it requires to build different HMMs for each label which requires data-splitting. Hence these models are likely to be underfitting, leading to the poor result.

