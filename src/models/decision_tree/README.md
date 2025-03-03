## 1. Overview of Decision Tree Algorithm

### **What is Decision Tree?**
A **Decision Tree** is a supervised learning algorithm that splits data recursively based on feature conditions to make predictions.

### **How It Works:**
1. The algorithm selects the best feature to split data using **Entropy (Information Gain)**.
2. It recursively divides data until a stopping condition is met.
3. The final tree is used to classify new inputs.

### **Advantages**
- Simple to interpret and visualize.
- Handles both numerical and categorical data.
- Requires minimal data preprocessing.

### **Disadvantages**
- Prone to **overfitting**.

---
## 2. Model Performance

### **Results on Test Data**
- Accuracy: 0.60
- F1-score: 0.60


### **Confusion Matrix Analysis**
| **Actual / Predicted** | **0 (Negative)** | **1 (Neutral)** | **2 (Positive)** |
|------------------------|-----------------|-----------------|-----------------|
| **0 (Negative)**       | **53** (Correct) |  44 (Misclassified) |  3 (Misclassified) |
| **1 (Neutral)**        |  27 (Misclassified) | **90** (Correct) |  26 (Misclassified) |
| **2 (Positive)**       |  10 (Misclassified) |  31 (Misclassified) | **70** (Correct) |

### **Key Observations**
- **Negative reviews (0) are often confused with Neutral (44 samples).**  
- **Neutral reviews (1) have the highest accuracy but still misclassified as Negative (27) or Positive (26).**  
- **Positive reviews (2) are sometimes confused with Neutral (31 samples).**  

---

