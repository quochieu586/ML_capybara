
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn import tree
from data import preprocess,feature_extraction
import matplotlib.pyplot as plt
import numpy as np

# Preprocessing dataset
pre_proc = preprocess.Preprocessing()
data = pre_proc.read_CSV('test.csv')

data['processed_text'] = data['text'].apply(pre_proc.preprocess)
#print(data.info())

# # Splitting the dataset into train and test
X = data['processed_text']
y = data['sentiment']
feature_extract = feature_extraction.FeatureExtraction()
X_train,X_val, X_test, y_train,y_val, y_test = feature_extract.split_dataset(X,y)
#print(f"Number of training samples: {len(X_train)}")


# # Extracting the features from the text data using TF-IDF
vectorizer = TfidfVectorizer(ngram_range=(1,3), max_features=50000)
X_train_vectors = vectorizer.fit_transform(X_train)
X_test_vectors = vectorizer.transform(X_test)

#Training with the decision tree
dt_classifier = DecisionTreeClassifier()
dt_classifier.fit(X_train_vectors, y_train)

# Making predictions on the test data
y_pred = dt_classifier.predict(X_test_vectors)

# Evaluating the model performance
print("Accuracy: ",accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))



from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
cm = confusion_matrix(y_test, y_pred)

# Hiển thị ma trận
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues')
plt.show()