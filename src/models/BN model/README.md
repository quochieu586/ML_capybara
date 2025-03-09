# **Text Classification Model using Bayesian Network**

## **Overview**
This project focuses on building a text classification model using a **Bayesian Network** for sentiment classification. The model is designed to classify text data into predefined sentiment categories: **negative**, **neutral**, and **positive**.

## **Key Features**
- **Text Preprocessing**: Includes tokenization, stopword removal, and lemmatization for cleaning text data. A list of token is returned after the preprocessing step in this model rather than a sentence.
- **Bayesian Network Model**: Utilizes a probabilistic graphical model to predict sentiment classes. However the bayesian network that would be built is under the Naive Bayes assumption, meaning each word is independent of each other when classifying the sentiment. 
- **Conditional Probability Distributions (CPDs)**: Defines the probability relationships between words and sentiment.

## **Model Architecture**
- **Graph Structure**: The Bayesian Network is defined with edges connecting sentiment labels to each word in the vocabulary of the corpus.
- **Conditional Probability Distributions (CPDs)**: Probabilities are learned based on word occurrences in sentiment classes.
- **Inference Mechanism**: Uses variable elimination to determine the most likely sentiment given observed words, which voids redundant computations and exploits the network structure to make inference more efficient instead of brute forcing all possible cases.

## **Evaluation Metrics**
The model's performance was evaluated using:
1. **F1-Score**: A weighted average of precision and recall.
2. **AUC-ROC**: Measures the model’s ability to distinguish between classes.

### **Confusion Matrix**
- **Final F1-Score**: 0.55
- **Final Accuracy**: 0.5537

### **AUC-ROC Scores**
- **Negative Sentiment:** **0.80** (Good separation)
- **Positive Sentiment:** **0.78** (Good separation)
- **Neutral Sentiment:** **0.64** (Decent separation)
- **Macro-Averaged ROC AUC Score**: 0.7400
- **Weighted ROC AUC Score:** 0.7289
- 
## **Model Strengths**
1. **AUC-ROC Performance**: The model display a strong classification capability for strong sentiment with **AUC-ROC score of 0.80 for Negative Sentiment** and **0.78 for Positive Sentiment**.
2. **Flexibility**: The Bayesian structure allows easy expansion by adding more words or sentiment classes just by modifying the corpus.

## **Model Weaknesses and Areas for Improvement**
1. **Accuracy**: The model’s F1-score is **55%**, which shows an above average performace in gerneral.
2. **Neutral Class Struggles**: The **AUC-ROC score of 0.64** for Neutral Sentiment has some difficulty in picking out neutral samples.
3. **Data Representation**: The model currently relies on word counts to calculate the probabilities rather the using well built libraries like **nltk**.
4. **Running Time**: The model's network can be built quickly in seconds. However, evaluating the models requires thoudsand of validation sample resulting in a running of 12-15 minutes.

## **Analysis and Next Steps**
- **Better Proprocessing technique**: The current hard-coded preprocessing procedure has yet to taken special cases into accounts such as website links or stylized word such as "lazyyyyyyyyy", that causes the model to treat them as a distinct word in the vocabulary. Since the dataset that this model was trained on was fromed by collecting the tweets from Twitter, a non-negligible amount of these special words aren't contributing to the classification task.

## **Conclusion**
This **Bayesian Network** provides an reasonable baseline for sentiment classification. The model performs reasonably well for **Negative and Positive** sentiment but struggles with **Neutral** classification. Future improvements could involve handling the special word from user's tweet in the preprocessing step by researching the wording style of internet users.
