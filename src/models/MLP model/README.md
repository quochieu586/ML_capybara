# Text Classification Model using Complex MLP

## Overview

This project focuses on building a text classification model using a **Complex MLP (Multilayer Perceptron)** architecture. The model is designed to classify text data into predefined sentiment categories, specifically **negative**, **neutral**, and **positive**. It utilizes natural language processing (NLP) techniques such as text preprocessing and TF-IDF vectorization for feature extraction.

## Key Features:
- **Text Preprocessing**: The model includes preprocessing steps such as tokenization, stopword removal, and lemmatization to clean and prepare the text data.
- **TF-IDF Vectorization**: Term Frequency-Inverse Document Frequency (TF-IDF) is used to convert the text data into numerical features.
- **Complex MLP Architecture**: A deep neural network with multiple hidden layers, batch normalization, Leaky ReLU activation, and dropout regularization is used to model the relationships in the text data.

## Model Architecture:
- **Input Layer**: The input layer is the feature vector from TF-IDF representation of the text data.
- **Hidden Layers**: Three fully connected hidden layers with Leaky ReLU activations and batch normalization are used to capture complex patterns in the data.
- **Output Layer**: The output layer uses a softmax activation function to predict the probability of each sentiment class (negative, neutral, or positive).

## Hyperparameter Tuning
The model's hyperparameters were optimized using **Optuna**, a hyperparameter optimization framework, to find the best configuration for **dropout rate** and **L2 regularization**. The model was trained for 15 epochs to ensure convergence.

## Evaluation Metrics
The model's performance was evaluated using the following metrics:
1. **Accuracy**: Measures the percentage of correctly predicted labels out of all predictions.
2. **F1-Score**: A weighted average of precision and recall, useful for imbalanced datasets.
3. **AUC-ROC**: The Area Under the Receiver Operating Characteristic Curve, which evaluates the model's ability to distinguish between classes.

### Results:
- **Accuracy**: 0.5059
- **F1-Score**: 0.3782
- **AUC-ROC**:
    + Negative: 0.7498
    + Neural: 0.6177
    + Positive: 0.8682

## Model Strengths
1. **AUC-ROC**: The model performs relatively well with AUC-ROC scores for each model. This indicates that the model is capable of distinguishing between classes in the dataset. The AUC-ROC value suggests a good trade-off between the true positive rate (sensitivity) and the false positive rate (1-specificity).
   
2. **Flexibility**: The model uses a flexible architecture that can be adapted to other classification tasks by changing the input features and output layers.

3. **Hyperparameter Optimization**: The model was fine-tuned using Optuna, which helps in optimizing the key hyperparameters and improving the model's performance over random search or manual tuning.

## Model Weaknesses and Areas for Improvement
1. **Accuracy**: The final accuracy of the model is relatively low at **40.40%**, indicating that the model has not fully learned to predict the correct classes. This could be due to various factors, including insufficient training data, suboptimal architecture, or noisy features.
   
2. **F1-Score**: The **F1-score** of 0.1918 is also quite low. This suggests that while the model might be able to predict some of the classes correctly, it has a high rate of false positives or false negatives, which impacts the precision and recall.

3. **Class Imbalance**: The model's performance may be affected by class imbalance, where one class (likely the neutral class) dominates the training data. If class distribution is heavily skewed, the model may have difficulty learning minority classes.

## Analysis and Next Steps
- **Addressing Class Imbalance**: The performance could be improved by using techniques such as **SMOTE (Synthetic Minority Over-sampling Technique)** for oversampling minority classes or **undersampling** the majority class.
  
- **Model Improvements**: 
    - **Architecture Enhancements**: The current architecture may be too simple to capture complex relationships. Adding more layers, using advanced architectures like **LSTM** or **BERT**, or exploring pre-trained embeddings like **Word2Vec** or **GloVe** could improve performance.
    - **Additional Regularization**: Exploring further regularization techniques like **early stopping** or **gradient clipping** may help to reduce overfitting and improve generalization.
  
- **Data Augmentation**: Using more diverse data for training or augmenting the dataset with techniques like paraphrasing could improve the model's robustness.

## Conclusion
This text classification model using a Complex MLP architecture provides a baseline for sentiment analysis tasks. While the model's performance (especially in terms of accuracy and F1-score) needs improvement, the **AUC-ROC** score indicates that the model is able to differentiate between classes. Further experimentation with data preprocessing, model architecture, and hyperparameter tuning could lead to significant improvements in classification accuracy.
