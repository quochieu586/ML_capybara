
# Text Classification Model using Linear SVM and Kernel SVM

## Text Classification Model using Linear SVM

### Overview

This project focuses on building a text classification model using a **Linear SVM (Support Vector Machine)** architecture. The model is designed to classify text data into predefined sentiment categories: **negative**, **neutral**, and **positive**. It leverages natural language processing (NLP) techniques such as text preprocessing, TF-IDF vectorization for feature extraction, and a custom PCA implementation for dimensionality reduction.

### Key Features:
- **Text Preprocessing**: Includes tokenization, stopword removal, and lemmatization to clean and prepare the text data.
- **TF-IDF Vectorization**: Term Frequency-Inverse Document Frequency (TF-IDF) converts text data into numerical features, with bigrams (`ngram_range=(1, 2)`) and a minimum document frequency (`min_df=5`) for improved feature quality.
- **Custom PCA**: A custom Principal Component Analysis (PCA) implementation reduces the dimensionality of TF-IDF features while retaining 90% of the variance.
- **Linear SVM with OvR**: A custom `OvRLinearSVM` class implements a multi-class SVM using the One-vs-Rest (OvR) strategy, with a linear decision boundary to classify sentiments.

### Model Architecture:
- **Input Layer**: The input is the TF-IDF feature vector, reduced to a lower-dimensional space using custom PCA.
- **Linear SVM**: Uses a linear kernel to find the optimal hyperplane that separates classes, with hinge loss and L2 regularization (`lambda_param=0.001`).
- **OvR Strategy**: Trains one binary SVM classifier per class (negative, neutral, positive), predicting the class with the highest decision score.
- **Output**: The model outputs the predicted sentiment class (negative, neutral, or positive) for each input text.

### Hyperparameters
- **SVM Parameters**:
  - Learning rate (`learning_rate`): 0.01
  - Regularization parameter (`lambda_param`): 0.001
  - Number of iterations (`n_iters`): 5,000
- **PCA**: Custom PCA retains 90% of the variance, resulting in 2,878 components.
- The model was trained on 85% of the `train.csv` data (23,358 samples) and validated on 15% (4,122 samples). The test set (`test.csv`) contains 3,534 samples.

### Evaluation Metrics
The model's performance was evaluated on both the validation and test sets using the following metrics:
1. **Accuracy**: Measures the percentage of correctly predicted labels out of all predictions.
2. **F1-Score**: A weighted average of precision and recall, useful for assessing performance on imbalanced datasets.
3. **Recall**: The proportion of true positives correctly identified, averaged across classes (macro average).
4. **AUC-ROC**: The Area Under the Receiver Operating Characteristic Curve, evaluating the model's ability to distinguish between classes.

#### Results:
- **Validation Set** (4,122 samples):
  - **Accuracy**: 0.6249 (62.49%)
  - **F1-Score**: 0.6254 (macro average)
  - **Recall**: 0.62 (macro average)
  - **AUC-ROC**: Not explicitly computed, but decision scores were used to plot ROC-AUC curves for each class (negative, neutral, positive).
  - **Class-wise Metrics**:
    - **Negative** (1,167 samples): Precision: 0.61, Recall: 0.61, F1-Score: 0.61
    - **Neutral** (1,668 samples): Precision: 0.58, Recall: 0.67, F1-Score: 0.62
    - **Positive** (1,287 samples): Precision: 0.73, Recall: 0.57, F1-Score: 0.64

- **Test Set** (3,534 samples):
  - **Accuracy**: 0.6488 (64.88%)
  - **F1-Score**: 0.65 (macro average)
  - **Recall**: 0.64 (macro average)
  - **AUC-ROC**: Not explicitly computed, but decision scores were used to plot ROC-AUC curves for each class (negative, neutral, positive).
  - **Class-wise Metrics**:
    - **Negative** (1,001 samples): Precision: 0.63, Recall: 0.61, F1-Score: 0.62
    - **Neutral** (1,430 samples): Precision: 0.60, Recall: 0.70, F1-Score: 0.65
    - **Positive** (1,103 samples): Precision: 0.76, Recall: 0.61, F1-Score: 0.68

### Model Strengths
1. **Balanced Performance**: The model achieves balanced F1-scores (validation: 0.6254, test: 0.65 macro average) across all classes, indicating improved handling of class imbalance (neutral: 11,117, positive: 8,582, negative: 7,781) compared to previous custom implementations (e.g., 28.3% accuracy with heavy bias toward `negative`).
   
2. **Custom PCA**: The custom PCA implementation retains 90% of the variance (explained variance ratio: 0.9000) with 2,878 components, preserving most of the meaningful information in the TF-IDF features while reducing dimensionality from 5,000 features.

3. **Class Weighting**: The model uses class weights (negative: 1.177, neutral: 0.824, positive: 1.067) to address class imbalance, improving predictions for minority classes (`negative`, `positive`).

4. **Score Normalization**: Decision scores are normalized using z-score normalization, ensuring balanced predictions across classes in the OvR strategy.

5. **Generalization**: The test accuracy (64.88%) is slightly higher than the validation accuracy (62.49%), suggesting the model generalizes well to unseen data.

### Model Weaknesses and Areas for Improvement
1. **Accuracy**: The accuracy (validation: 62.49%, test: 64.88%) is an improvement over earlier custom implementations (e.g., 28.3%), but it is lower than both the custom `KernelSVM` (validation: 68.87%, test: 69.78%) and `scikit-learn`’s `SVC` (previously ~0.72). This suggests that the linear kernel may not capture complex, non-linear relationships in the data as effectively as an RBF kernel.

2. **Class-wise Performance**:
   - The `positive` class has high precision (validation: 0.73, test: 0.76) but lower recall (validation: 0.57, test: 0.61), indicating that the model misses some `positive` instances (false negatives).
   - The `neutral` class has lower precision (validation: 0.58, test: 0.60) compared to `negative` (validation: 0.61, test: 0.63) and `positive`, suggesting a higher rate of false positives for `neutral` predictions.

3. **Linear Kernel Limitation**: The linear kernel assumes that the data is linearly separable in the reduced feature space, which may not hold true for text data with complex, non-linear relationships. This could explain the performance gap compared to the `KernelSVM` with an RBF kernel.

4. **Computational Efficiency**: The custom PCA implementation, while effective, may be slower than `scikit-learn`’s optimized PCA due to the use of eigenvalue decomposition, especially for large datasets (27,480 samples in `train.csv`).

### Analysis and Next Steps
- **Addressing Class Imbalance**: Although class weights have mitigated imbalance, techniques like **SMOTE** (Synthetic Minority Over-sampling Technique) or **undersampling** the majority class (`neutral`) could further improve performance for minority classes (`negative`, `positive`).

- **Model Improvements**:
  - **Switch to Non-Linear Kernels**: Using a kernel SVM (e.g., RBF kernel) or reverting to `scikit-learn`’s `SVC` with an RBF kernel could better capture non-linear relationships, as demonstrated by the `KernelSVM`’s higher accuracy (test: 69.78%) and `SVC`’s (~0.72).
  - **Hyperparameter Tuning**: Experiment with different values of `learning_rate` (e.g., [0.001, 0.1]), `lambda_param` (e.g., [0.0001, 0.01]), and `n_iters` (e.g., [5000, 10000]) to improve convergence.
  - **Advanced Architectures**: Replace the SVM with a deep learning model (e.g., LSTM, BERT) or use pre-trained embeddings (e.g., Word2Vec, GloVe) to capture semantic relationships in the text.

- **Optimize Custom PCA**:
  - Use Singular Value Decomposition (SVD) instead of eigenvalue decomposition for better numerical stability and speed.
  - Implement sparse PCA to handle TF-IDF features without dense matrix conversion, reducing memory usage and computation time.

- **Feature Engineering**:
  - Increase `max_features` (e.g., 10,000) in `TfidfVectorizer` or use custom stopwords to retain sentiment-related words (e.g., “not”, “very”).
  - Explore word embeddings or contextual embeddings (e.g., BERT) for richer feature representations.

- **Data Augmentation**: Augment the dataset with techniques like paraphrasing or back-translation to improve the model’s robustness, especially for underrepresented classes.

## Text Classification Model using Kernel SVM

### Overview

This project focuses on building a text classification model using a **Kernel SVM (Support Vector Machine)** architecture with a radial basis function (RBF) kernel. The model is designed to classify text data into predefined sentiment categories: **negative**, **neutral**, and **positive**. It leverages natural language processing (NLP) techniques such as text preprocessing, TF-IDF vectorization for feature extraction, and a custom PCA implementation for dimensionality reduction.

### Key Features:
- **Text Preprocessing**: Includes tokenization, stopword removal, and lemmatization to clean and prepare the text data.
- **TF-IDF Vectorization**: Term Frequency-Inverse Document Frequency (TF-IDF) converts text data into numerical features, with bigrams (`ngram_range=(1, 2)`) and a minimum document frequency (`min_df=5`) for improved feature quality.
- **Custom PCA**: A custom Principal Component Analysis (PCA) implementation reduces the dimensionality of TF-IDF features while retaining 90% of the variance.
- **Kernel SVM with OvR**: A custom `OvRKernelSVM` class implements a multi-class SVM using the One-vs-Rest (OvR) strategy, with an RBF kernel for capturing non-linear relationships in the data.

### Model Architecture:
- **Input Layer**: The input is the TF-IDF feature vector, reduced to a lower-dimensional space using custom PCA.
- **Kernel SVM**: Utilizes an RBF kernel (`gamma=0.01`) to map the data into a higher-dimensional space, where a linear boundary can separate the classes.
- **OvR Strategy**: Trains one binary SVM classifier per class (negative, neutral, positive), predicting the class with the highest decision score.
- **Output**: The model outputs the predicted sentiment class (negative, neutral, or positive) for each input text.

### Hyperparameters
- **SVM Parameters**: 
  - Regularization parameter (`C`): 1.0
  - Kernel: RBF with `gamma=0.01`
  - Maximum iterations (`max_iter`): 50
  - Tolerance (`tol`): 1e-3
- **PCA**: Custom PCA retains 90% of the variance, resulting in 2,878 components.
- The model was trained on 85% of the `train.csv` data (23,358 samples) and validated on 15% (4,122 samples). The test set (`test.csv`) contains 3,534 samples.

### Evaluation Metrics
The model's performance was evaluated on both the validation and test sets using the following metrics:
1. **Accuracy**: Measures the percentage of correctly predicted labels out of all predictions.
2. **F1-Score**: A weighted average of precision and recall, useful for assessing performance on imbalanced datasets.
3. **Recall**: The proportion of true positives correctly identified, averaged across classes (macro average).
4. **AUC-ROC**: The Area Under the Receiver Operating Characteristic Curve, evaluating the model's ability to distinguish between classes.

#### Results:
- **Validation Set** (4,122 samples):
  - **Accuracy**: 0.6887 (68.87%)
  - **F1-Score**: 0.6912 (macro average)
  - **Recall**: 0.69 (macro average)
  - **AUC-ROC**: Not explicitly computed, but decision scores were used to plot ROC-AUC curves for each class (negative, neutral, positive).
  - **Class-wise Metrics**:
    - **Negative** (1,167 samples): Precision: 0.70, Recall: 0.65, F1-Score: 0.67
    - **Neutral** (1,668 samples): Precision: 0.63, Recall: 0.71, F1-Score: 0.67
    - **Positive** (1,287 samples): Precision: 0.77, Recall: 0.70, F1-Score: 0.73

- **Test Set** (3,534 samples):
  - **Accuracy**: 0.6978 (69.78%)
  - **F1-Score**: 0.7002 (macro average)
  - **Recall**: 0.70 (macro average)
  - **AUC-ROC**: Not explicitly computed, but decision scores were used to plot ROC-AUC curves for each class (negative, neutral, positive).
  - **Class-wise Metrics**:
    - **Negative** (1,001 samples): Precision: 0.68, Recall: 0.66, F1-Score: 0.67
    - **Neutral** (1,430 samples): Precision: 0.65, Recall: 0.70, F1-Score: 0.68
    - **Positive** (1,103 samples): Precision: 0.78, Recall: 0.72, F1-Score: 0.75

### Model Strengths
1. **Balanced Performance**: The model achieves balanced F1-scores (validation: 0.6912, test: 0.7002 macro average) across all classes, indicating improved handling of class imbalance (neutral: 11,117, positive: 8,582, negative: 7,781) compared to previous custom implementations (e.g., 28.3% accuracy with heavy bias toward `negative`).
   
2. **Custom PCA**: The custom PCA implementation retains 90% of the variance (explained variance ratio: 0.9000) with 2,878 components, ensuring that most of the meaningful information in the TF-IDF features is preserved while reducing dimensionality from 5,000 features.

3. **Class Weighting**: The model uses class weights (negative: 1.177, neutral: 0.824, positive: 1.067) to address class imbalance, improving predictions for minority classes (`negative`, `positive`).

4. **Generalization**: The test accuracy (69.78%) is slightly higher than the validation accuracy (68.87%), suggesting the model generalizes well to unseen data.

5. **Non-Linear Kernel**: The RBF kernel allows the model to capture complex, non-linear relationships in the data, outperforming the `LinearSVM` (test accuracy: 64.88%).

### Model Weaknesses and Areas for Improvement
1. **Accuracy**: While the accuracy (validation: 68.87%, test: 69.78%) is a significant improvement over earlier custom implementations (e.g., 28.3%), it is still lower than `scikit-learn`’s `SVC` (previously ~0.72). This suggests that the custom `OvRKernelSVM` implementation may not converge as effectively as optimized library implementations.

2. **Class-wise Performance**:
   - The `neutral` class has lower precision (validation: 0.63, test: 0.65) compared to `negative` (validation: 0.70, test: 0.68) and `positive` (validation: 0.77, test: 0.78), indicating a higher rate of false positives for `neutral` predictions.
   - The recall for `negative` (validation: 0.65, test: 0.66) is lower than for `neutral` (validation: 0.71, test: 0.70) and `positive` (validation: 0.70, test: 0.72), suggesting the model misses some `negative` instances.

3. **Computational Efficiency**: The custom PCA implementation, while transparent, may be slower than `scikit-learn`’s optimized PCA due to the use of eigenvalue decomposition. For large datasets (27,480 samples in `train.csv`), this can impact training time.

4. **Feature Quality**: The TF-IDF features with `max_features=5000` may still include noise or miss important sentiment cues, limiting the model’s ability to achieve higher accuracy.

### Analysis and Next Steps
- **Addressing Class Imbalance**: While class weights have improved balance, further techniques like **SMOTE** (Synthetic Minority Over-sampling Technique) or **undersampling** the majority class (`neutral`) could enhance performance for minority classes (`negative`, `positive`).

- **Model Improvements**:
  - **Switch to `scikit-learn`’s `SVC`**: Given the performance gap (test: 69.78% vs. ~0.72 with `SVC`), using `scikit-learn`’s `SVC` with the same custom PCA could yield better results due to its optimized implementation.
  - **Hyperparameter Tuning**: Experiment with different values of `C` (e.g., [0.1, 1.0, 10.0]), `gamma` (e.g., [0.001, 0.01, 0.1]), and `max_iter` (e.g., [50, 100]) to improve convergence.
  - **Advanced Architectures**: Replace the SVM with a deep learning model (e.g., LSTM, BERT) or use pre-trained embeddings (e.g., Word2Vec, GloVe) to capture semantic relationships in the text.

- **Optimize Custom PCA**:
  - Use Singular Value Decomposition (SVD) instead of eigenvalue decomposition for better numerical stability and speed.
  - Implement sparse PCA to handle TF-IDF features without dense matrix conversion, reducing memory usage and computation time.

- **Feature Engineering**:
  - Increase `max_features` (e.g., 10,000) in `TfidfVectorizer` or use custom stopwords to retain sentiment-related words (e.g., “not”, “very”).
  - Explore word embeddings or contextual embeddings (e.g., BERT) for richer feature representations.

- **Data Augmentation**: Augment the dataset with techniques like paraphrasing or back-translation to improve the model’s robustness, especially for underrepresented classes.

## Conclusion

This text classification model using a custom Kernel SVM with an RBF kernel and OvR strategy provides a solid foundation for sentiment analysis, achieving a validation accuracy of 68.87% and a test accuracy of 69.78%, with macro F1-scores of 0.6912 and 0.7002, respectively. The custom PCA implementation effectively retains 90% of the variance, and class weighting helps address class imbalance. The model’s performance is better than the `LinearSVM` (test accuracy: 64.88%), thanks to the RBF kernel’s ability to capture non-linear relationships, but it still falls short of `scikit-learn`’s `SVC` (~0.72). The slight improvement in test accuracy over validation suggests good generalization, but future work should focus on hyperparameter tuning, advanced feature engineering, and potentially switching to optimized library implementations like `SVC` to achieve higher accuracy and efficiency.

## Performance Comparison Table

The table below compares the performance of the `LinearSVM` and `KernelSVM` models on the validation and test sets in terms of accuracy, F1-score (macro average), and recall (macro average).

| **Model**    | **Set**    | **Accuracy** | **F1-Score** | **Recall** |
|--------------|------------|--------------|--------------|------------|
| LinearSVM    | Validation | 62.49%       | 0.6254       | 0.62       |
| LinearSVM    | Test       | 64.88%       | 0.65         | 0.64       |
| KernelSVM    | Validation | 68.87%       | 0.6912       | 0.69       |
| KernelSVM    | Test       | 69.78%       | 0.7002       | 0.70       |

---

### Notes on the Updated Output

- **Test Set Performance**: The `KernelSVM` test accuracy (69.78%) is slightly higher than the validation accuracy (68.87%), indicating good generalization. The F1-score (0.7002 macro average) and recall (0.70 macro average) also align with validation, showing consistent performance.
- **Comparison with `LinearSVM`**: The `KernelSVM` outperforms the `LinearSVM` in all metrics (test accuracy: 69.78% vs. 64.88%, F1-score: 0.7002 vs. 0.65, recall: 0.70 vs. 0.64), confirming the advantage of the RBF kernel in capturing non-linear relationships.
- **Table Clarity**: The table provides a clear side-by-side comparison, making it easy to see that `KernelSVM` consistently performs better than `LinearSVM` on both validation and test sets.