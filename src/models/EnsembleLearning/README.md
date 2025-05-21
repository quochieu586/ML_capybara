# **Text Classification Model using Gradient Boosting**

## **Overview**
This project implements a Gradient Boosting approach for text classification, specifically focusing on sentiment analysis. The model combines multiple decision trees in a sequential manner to improve classification performance.

## **Key Features**
- **Text Preprocessing**: Utilizes tokenization, stopword removal, and lemmatization for text cleaning
- **Feature Extraction**: Implements TF-IDF vectorization with configurable n-gram ranges and feature limits
- **Gradient Boosting**: Custom implementation of multiclass gradient boosting with decision trees as base learners

## **Model Architecture**
- **Gradient Boosting Implementation**:
  - Custom `MulticlassGradientBoosting` class
  - Supports multiple classes through one-hot encoding
  - Uses decision trees as base learners
  - Implements softmax for probability estimation
  - Sequential training of trees to minimize residual errors
  - Learning rate control for model stability

## **Evaluation Metrics**
The model's performance is evaluated using:
1. **Accuracy**: Overall classification accuracy on the test set
2. **F1-Score**: Weighted average of precision and recall
3. **Confusion Matrix**: Detailed breakdown of classification results

### **Confusion Matrix**

|               | Predicted: negative | Predicted: neutral | Predicted: positive |
|---------------|--------------------|--------------------|---------------------|
| **True: negative** | 488                | 425                | 88                  |
| **True: neutral**  | 207                | 1039                | 184                 |
| **True: positive** | 71                | 367                | 665                 |

- **F1 Score (weighted):** 0.6178
- **Accuracy :** 0.6203
- **AUC-ROC :**
  + Negative: 0.8044
  + Neural: 0.7404
  + Positive: 0.8350

## **Model Strengths**
1. **Consistent Performance Across Classes**: The model achieves a reasonable balance in classifying all three sentiment classes, with the highest accuracy for the neutral class (1039 correct predictions).
2. **Robustness to Class Imbalance**: Despite class imbalance, the model maintains moderate performance across all classes, as reflected in the confusion matrix and F1 score.
3. **Customizability**: The model allows for tuning of key parameters (learning rate, number of trees, tree depth) to further improve performance.

## **Model Weaknesses and Areas for Improvement**
1. **Confusion Between Classes**: There is notable confusion between negative and neutral classes (425 negative samples predicted as neutral).
2. **Moderate F1 Score**: An F1 score of 0.6203 suggests the model can be further optimized for better precision and recall.
3. **Computational Complexity**: Training multiple trees sequentially can be computationally expensive.


## **Analysis and Next Steps**
- **Enhance Feature Engineering**: Explore additional text features or embeddings to help the model better distinguish between similar classes, especially negative and neutral.
- **Parameter Tuning**: Further optimize hyperparameters (learning rate, number of estimators, tree depth) to improve F1 score and reduce class confusion.

## **Conclusion**
The Gradient Boosting approach provides a solid baseline for sentiment classification, achieving a weighted F1 score of 0.6203. While the model performs best on the neutral class, there is room for improvement in distinguishing between negative and neutral sentiments. Future work should be focusing on using other inputs such as embedding vectors rather than tf-idf to improve performance.