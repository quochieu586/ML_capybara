# Text Classification Model using Linear SVM

## Overview

This project focuses on building a text classification model using a **Linear SVM (Support Vector Machine)** architecture. The model is designed to classify text data into predefined sentiment categories: **negative**, **neutral**, and **positive**. It leverages natural language processing (NLP) techniques such as text preprocessing, TF-IDF vectorization for feature extraction, and a custom PCA implementation for dimensionality reduction.

## Key Features:
- **Text Preprocessing**: Includes tokenization, stopword removal, and lemmatization to clean and prepare the text data.
- **TF-IDF Vectorization**: Term Frequency-Inverse Document Frequency (TF-IDF) converts text data into numerical features, with bigrams (`ngram_range=(1, 2)`) and a minimum document frequency (`min_df=5`) for improved feature quality.
- **Custom PCA**: A custom Principal Component Analysis (PCA) implementation reduces the dimensionality of TF-IDF features while retaining 90% of the variance.
- **Linear SVM with OvR**: A custom `OvRLinearSVM` class implements a multi-class SVM using the One-vs-Rest (OvR) strategy, with a linear decision boundary to classify sentiments.

## Model Architecture:
- **Input Layer**: The input is the TF-IDF feature vector, reduced to a lower-dimensional space using custom PCA.
- **Linear SVM**: Uses a linear kernel to find the optimal hyperplane that separates classes, with hinge loss and L2 regularization (`lambda_param=0.001`).
- **OvR Strategy**: Trains one binary SVM classifier per class (negative, neutral, positive), predicting the class with the highest decision score.
- **Output**: The model outputs the predicted sentiment class (negative, neutral, or positive) for each input text.

## Hyperparameters
- **SVM Parameters**:
  - Learning rate (`learning_rate`): 0.01
  - Regularization parameter (`lambda_param`): 0.001
  - Number of iterations (`n_iters`): 5,000
- **PCA**: Custom PCA retains 90% of the variance, resulting in 2,878 components.
- The model was trained on 85% of the `train.csv` data (23,358 samples) and validated on 15% (4,122 samples).

## Evaluation Metrics
The model's performance was evaluated using the following metrics:
1. **Accuracy**: Measures the percentage of correctly predicted labels out of all predictions.
2. **F1-Score**: A weighted average of precision and recall, useful for assessing performance on imbalanced datasets.
3. **AUC-ROC**: The Area Under the Receiver Operating Characteristic Curve, evaluating the model's ability to distinguish between classes.

### Results:
- **Accuracy**: 0.6249 (62.49%)
- **F1-Score**: 0.6254 (macro average)
- **AUC-ROC**: Not explicitly computed in the provided output, but decision scores were used to plot ROC-AUC curves for each class (negative, neutral, positive).
- **Class-wise Metrics** (validation set, 4,122 samples):
  - **Negative** (1,167 samples): Precision: 0.61, Recall: 0.61, F1-Score: 0.61
  - **Neutral** (1,668 samples): Precision: 0.58, Recall: 0.67, F1-Score: 0.62
  - **Positive** (1,287 samples): Precision: 0.73, Recall: 0.57, F1-Score: 0.64

## Model Strengths
1. **Balanced Performance**: The model achieves a balanced F1-score (0.6254 macro average) across all classes, indicating improved handling of class imbalance (neutral: 11,117, positive: 8,582, negative: 7,781) compared to previous custom implementations (e.g., 28.3% accuracy with heavy bias toward `negative`).
   
2. **Custom PCA**: The custom PCA implementation retains 90% of the variance (explained variance ratio: 0.9000) with 2,878 components, preserving most of the meaningful information in the TF-IDF features while reducing dimensionality from 5,000 features.

3. **Class Weighting**: The model uses class weights (negative: 1.177, neutral: 0.824, positive: 1.067) to address class imbalance, improving predictions for minority classes (`negative`, `positive`).

4. **Score Normalization**: Decision scores are normalized using z-score normalization, ensuring balanced predictions across classes in the OvR strategy.

## Model Weaknesses and Areas for Improvement
1. **Accuracy**: The accuracy of 62.49% is an improvement over earlier custom implementations (e.g., 28.3%), but it is lower than both the custom `KernelSVM` (68.87%) and `scikit-learn`’s `SVC` (previously ~0.72). This suggests that the linear kernel may not capture complex, non-linear relationships in the data as effectively as an RBF kernel.

2. **Class-wise Performance**:
   - The `positive` class has a high precision (0.73) but a lower recall (0.57), indicating that the model misses some `positive` instances (false negatives).
   - The `neutral` class has a lower precision (0.58) compared to `negative` (0.61) and `positive` (0.73), suggesting a higher rate of false positives for `neutral` predictions.

3. **Linear Kernel Limitation**: The linear kernel assumes that the data is linearly separable in the reduced feature space, which may not hold true for text data with complex, non-linear relationships. This could explain the performance gap compared to the `KernelSVM` with an RBF kernel.

4. **Computational Efficiency**: The custom PCA implementation, while effective, may be slower than `scikit-learn`’s optimized PCA due to the use of eigenvalue decomposition, especially for large datasets (27,480 samples).

## Analysis and Next Steps
- **Addressing Class Imbalance**: Although class weights have mitigated imbalance, techniques like **SMOTE** (Synthetic Minority Over-sampling Technique) or **undersampling** the majority class (`neutral`) could further improve performance for minority classes (`negative`, `positive`).

- **Model Improvements**:
  - **Switch to Non-Linear Kernels**: Using a kernel SVM (e.g., RBF kernel) or reverting to `scikit-learn`’s `SVC` with an RBF kernel could better capture non-linear relationships, as demonstrated by the `KernelSVM`’s higher accuracy (68.87%).
  - **Hyperparameter Tuning**: Experiment with different values of `learning_rate` (e.g., [0.001, 0.1]), `lambda_param` (e.g., [0.0001, 0.01]), and `n_iters` (e.g., [5000, 10000]) to improve convergence.
  - **Advanced Architectures**: Replace the SVM with a deep learning model (e.g., LSTM, BERT) or use pre-trained embeddings (e.g., Word2Vec, GloVe) to capture semantic relationships in the text.

- **Optimize Custom PCA**:
  - Use Singular Value Decomposition (SVD) instead of eigenvalue decomposition for better numerical stability and speed.
  - Implement sparse PCA to handle TF-IDF features without dense matrix conversion, reducing memory usage and computation time.

- **Feature Engineering**:
  - Increase `max_features` (e.g., 10,000) in `TfidfVectorizer` or use custom stopwords to retain sentiment-related words (e.g., “not”, “very”).
  - Explore word embeddings or contextual embeddings (e.g., BERT) for richer feature representations.

- **Data Augmentation**: Augment the dataset with techniques like paraphrasing or back-translation to improve the model’s robustness, especially for underrepresented classes.

## Conclusion

This text classification model using a custom Linear SVM with an OvR strategy provides a baseline for sentiment analysis, achieving a validation accuracy of 62.49% and a macro F1-score of 0.6254. The custom PCA implementation effectively retains 90% of the variance, and class weighting helps address class imbalance. However, the linear kernel limits the model’s ability to capture complex patterns, resulting in lower accuracy compared to the `KernelSVM` (68.87%) and `scikit-learn`’s `SVC` (~0.72). Future work should focus on adopting non-linear kernels, hyperparameter tuning, advanced feature engineering, and potentially switching to optimized library implementations like `SVC` to achieve higher accuracy and efficiency.

---

### Notes on the Output

- **Performance**: The accuracy (0.6249) and F1-score (0.6254) are improvements over the initial custom `KernelSVM` (28.3%), but the linear kernel underperforms compared to the RBF kernel `KernelSVM` (0.6887) due to its inability to capture non-linear relationships.
- **Custom PCA**: Retaining 90% variance with 2,878 components ensures most information is preserved, but the high number of components suggests sparse PCA or `TruncatedSVD` could be more efficient.
- **Class Distribution**: The dataset’s moderate imbalance (neutral: 11,117, positive: 8,582, negative: 7,781) is mitigated by class weights, but further balancing techniques could improve performance.

# Text Classification Model using Kernel SVM

## Overview

This project focuses on building a text classification model using a **Kernel SVM (Support Vector Machine)** architecture with a radial basis function (RBF) kernel. The model is designed to classify text data into predefined sentiment categories: **negative**, **neutral**, and **positive**. It leverages natural language processing (NLP) techniques such as text preprocessing, TF-IDF vectorization for feature extraction, and a custom PCA implementation for dimensionality reduction.

## Key Features:
- **Text Preprocessing**: Includes tokenization, stopword removal, and lemmatization to clean and prepare the text data.
- **TF-IDF Vectorization**: Term Frequency-Inverse Document Frequency (TF-IDF) converts text data into numerical features, with bigrams (`ngram_range=(1, 2)`) and a minimum document frequency (`min_df=5`) for improved feature quality.
- **Custom PCA**: A custom Principal Component Analysis (PCA) implementation reduces the dimensionality of TF-IDF features while retaining 90% of the variance.
- **Kernel SVM with OvR**: A custom `OvRKernelSVM` class implements a multi-class SVM using the One-vs-Rest (OvR) strategy, with an RBF kernel for capturing non-linear relationships in the data.

## Model Architecture:
- **Input Layer**: The input is the TF-IDF feature vector, reduced to a lower-dimensional space using custom PCA.
- **Kernel SVM**: Utilizes an RBF kernel (`gamma=0.01`) to map the data into a higher-dimensional space, where a linear boundary can separate the classes.
- **OvR Strategy**: Trains one binary SVM classifier per class (negative, neutral, positive), predicting the class with the highest decision score.
- **Output**: The model outputs the predicted sentiment class (negative, neutral, or positive) for each input text.

## Hyperparameters
- **SVM Parameters**: 
  - Regularization parameter (`C`): 1.0
  - Kernel: RBF with `gamma=0.01`
  - Maximum iterations (`max_iter`): 50
  - Tolerance (`tol`): 1e-3
- **PCA**: Custom PCA retains 90% of the variance, resulting in 2,878 components.
- The model was trained on 85% of the `train.csv` data (23,358 samples) and validated on 15% (4,122 samples).

## Evaluation Metrics
The model's performance was evaluated using the following metrics:
1. **Accuracy**: Measures the percentage of correctly predicted labels out of all predictions.
2. **F1-Score**: A weighted average of precision and recall, useful for assessing performance on imbalanced datasets.
3. **AUC-ROC**: The Area Under the Receiver Operating Characteristic Curve, evaluating the model's ability to distinguish between classes.

### Results:
- **Accuracy**: 0.6887 (68.87%)
- **F1-Score**: 0.6912 (macro average)
- **AUC-ROC**: Not explicitly computed in the provided output, but decision scores were used to plot ROC-AUC curves for each class (negative, neutral, positive).
- **Class-wise Metrics** (validation set, 4,122 samples):
  - **Negative** (1,167 samples): Precision: 0.70, Recall: 0.65, F1-Score: 0.67
  - **Neutral** (1,668 samples): Precision: 0.63, Recall: 0.71, F1-Score: 0.67
  - **Positive** (1,287 samples): Precision: 0.77, Recall: 0.70, F1-Score: 0.73

## Model Strengths
1. **Balanced Performance**: The model achieves a balanced F1-score (0.6912 macro average) across all classes, indicating improved handling of class imbalance (neutral: 11,117, positive: 8,582, negative: 7,781) compared to previous implementations (e.g., 28.3% accuracy with heavy bias toward `negative`).
   
2. **Custom PCA**: The custom PCA implementation retains 90% of the variance (explained variance ratio: 0.9000) with 2,878 components, ensuring that most of the meaningful information in the TF-IDF features is preserved while reducing dimensionality from 5,000 features.

3. **Flexibility**: The OvR strategy and custom PCA make the model adaptable to other classification tasks by adjusting the kernel, features, or number of components.

4. **Class Weighting**: The model uses class weights (negative: 1.177, neutral: 0.824, positive: 1.067) to address class imbalance, improving predictions for minority classes.

## Model Weaknesses and Areas for Improvement
1. **Accuracy**: While the accuracy of 68.87% is a significant improvement over previous custom implementations (e.g., 28.3%), it is still lower than `scikit-learn`’s `SVC` (previously ~0.72). This suggests that the custom `OvRLinearSVM` may not converge as effectively or capture complex patterns as well as optimized library implementations.

2. **Class-wise Performance**:
   - The `neutral` class has a lower precision (0.63) compared to `negative` (0.70) and `positive` (0.77), indicating a higher rate of false positives for `neutral` predictions.
   - The recall for `negative` (0.65) is lower than for `neutral` (0.71) and `positive` (0.70), suggesting the model misses some `negative` instances.

3. **Computational Efficiency**: The custom PCA implementation, while transparent, may be slower than `scikit-learn`’s optimized PCA due to the use of eigenvalue decomposition. For large datasets (27,480 samples), this can impact training time.

4. **Feature Quality**: The TF-IDF features with `max_features=5000` may still include noise or miss important sentiment cues, limiting the model’s ability to achieve higher accuracy.

## Analysis and Next Steps
- **Addressing Class Imbalance**: While class weights have improved balance, further techniques like **SMOTE** (Synthetic Minority Over-sampling Technique) or **undersampling** the majority class (`neutral`) could enhance performance for minority classes (`negative`, `positive`).

- **Model Improvements**:
  - **Switch to `scikit-learn`’s `SVC`**: Given the performance gap (0.6887 vs. ~0.72 with `SVC`), using `scikit-learn`’s `SVC` with the same custom PCA could yield better results due to its optimized implementation.
  - **Hyperparameter Tuning**: Experiment with different values of `learning_rate` (e.g., [0.001, 0.1]), `lambda_param` (e.g., [0.0001, 0.01]), and `n_iters` (e.g., [5000, 10000]) to improve convergence.
  - **Advanced Architectures**: Replace the linear SVM with a more complex model (e.g., LSTM, BERT) or use pre-trained embeddings (e.g., Word2Vec, GloVe) to capture semantic relationships in the text.

- **Optimize Custom PCA**:
  - Use Singular Value Decomposition (SVD) instead of eigenvalue decomposition for better numerical stability and speed.
  - Implement sparse PCA to handle TF-IDF features without dense matrix conversion.

- **Feature Engineering**:
  - Increase `max_features` (e.g., 10,000) in `TfidfVectorizer` or use custom stopwords to retain sentiment-related words (e.g., “not”, “very”).
  - Explore word embeddings or contextual embeddings (e.g., BERT) for richer feature representations.

- **Data Augmentation**: Augment the dataset with techniques like paraphrasing or back-translation to improve the model’s robustness, especially for underrepresented classes.

## Conclusion

This text classification model using a custom Kernel SVM with an RBF kernel and OvR strategy provides a solid baseline for sentiment analysis, achieving a validation accuracy of 68.87% and a macro F1-score of 0.6912. The custom PCA implementation effectively retains 90% of the variance, and class weighting helps address class imbalance. However, the model’s accuracy is still lower than `scikit-learn`’s `SVC`, and there is room for improvement in class-wise performance, particularly for the `neutral` class. Future work should focus on hyperparameter tuning, advanced feature engineering, and potentially switching to optimized library implementations like `SVC` to achieve higher accuracy and efficiency.

--- 

### Notes on the Output

- **Performance**: The accuracy (0.6887) and F1-score (0.6912) are improvements over the custom `KernelSVM` (28.3%), but the `neutral` class’s lower precision (0.63) indicates room for improvement.
- **Custom PCA**: Retaining 90% variance with 2,878 components is effective, but the high number of components suggests that sparse PCA or `TruncatedSVD` could be more efficient for TF-IDF features.
- **Class Distribution**: The dataset has a moderate imbalance (neutral: 11,117, positive: 8,582, negative: 7,781), which is mitigated by class weights but could benefit from further balancing techniques.