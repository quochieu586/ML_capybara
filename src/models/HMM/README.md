# Text Classification Model using Categorical HMM

## Overview

This project focuses on building a text classification model using a **Categorical Hidden Markov Model (CategoricalHMM)** architecture. The model is designed to classify text data into predefined sentiment categories, specifically **negative**, **neutral**, and **positive**. It leverages natural language processing (NLP) techniques such as text preprocessing and token-to-index encoding for feature extraction.

## Key Features:
- **Text Preprocessing**: Includes tokenization, stopword removal, and optional lemmatization to clean and standardize the text data.
- **Token Indexing**: Each word is converted into a categorical index to be used in the CategoricalHMM, which models sequences of discrete observations.
- **Categorical HMM Architecture**: Separate HMMs are trained for each sentiment class using sequences of token indices. During inference, the model computes the log-likelihood of a sample under each class-specific HMM and assigns the label of the highest likelihood.

## Model Architecture:
- **Observation Model**: Discrete observations using `CategoricalHMM` from the `hmmlearn` library.
- **Per-Class Modeling**: One HMM is trained per sentiment class.
- **Classification Strategy**: Classify by selecting the HMM model with the highest log-likelihood score for a given input sequence.

## Hyperparameter Tuning

Hyperparameters for the `CategoricalHMM` were tuned manually and semi-automatically with grid search strategies based on cross-validation performance. The following parameters were tuned:

- `n_components` (Number of hidden states):  
  Ranged from 2 to 10. Best results were obtained with **9 states**, balancing model complexity and generalization.

- `n_iter` (Number of EM iterations during fitting):  
  Set to **50** to ensure convergence without overfitting.

> Note: Due to the probabilistic nature of HMMs and variability in dataset quality, tuning was guided by validation set log-likelihood and macro F1-score across folds.

### Results:
- **Best number of components**: 9
- **Accuracy**: 0.4434
- **Classification Report**:
    - **Negative**: precision = 0.41, recall = 0.54, F1 = 0.46
    - **Neutral**: precision = 0.48, recall = 0.31, F1 = 0.38
    - **Positive**: precision = 0.45, recall = 0.53, F1 = 0.49
- **Macro Average F1-Score**: 0.44
- **Weighted Average F1-Score**: 0.44

## Model Strengths

1. **Sequence Awareness**: HMMs model word sequences explicitly, unlike bag-of-words or TF-IDF models.
2. **Class-specific Transition Patterns**: Training separate HMMs per class allows capturing unique linguistic patterns for each sentiment.
3. **Interpretability**: The transition and emission matrices offer interpretable insights into sequence behavior.
4. **Improved Class Separation**: Positive and negative classes achieved reasonable F1-scores, indicating useful pattern learning.

## Model Weaknesses and Areas for Improvement

1. **Neutral Class Performance**: The neutral class shows lower recall and F1-score, suggesting difficulty distinguishing from the other two classes.
2. **Overall Accuracy**: 44.34% indicates moderate performance; model may not generalize well to ambiguous or complex language structures.
3. **Markov Assumption Limitation**: HMMs assume limited memory, which restricts their ability to capture long-range dependencies in language.

## Analysis and Next Steps

- **Handle Class Imbalance**: Use techniques like SMOTE or stratified sampling to better balance the training data.
- **Explore Alternative Models**:
    - Conditional Random Fields (CRF)
    - Neural sequence models (e.g., LSTM, GRU, Transformer)
    - Hybrid approaches (e.g., embeddings + HMM)
- **Incorporate Pretrained Embeddings**: Integrate Word2Vec or GloVe to improve the representation of token inputs.
- **Data Augmentation**: Use paraphrasing or back-translation to diversify training examples.

## Conclusion

This project demonstrates the application of **Categorical Hidden Markov Models** to a multiclass sentiment classification problem. While the model shows promise in capturing sequential structure, especially for the positive and negative classes, further work is needed to improve its robustness and performance on more ambiguous or nuanced examples.