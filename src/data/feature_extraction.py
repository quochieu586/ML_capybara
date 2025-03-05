from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer
from sklearn.model_selection import train_test_split

class FeatureExtraction:
    def __init__(self, seed = 2025):
        """
            class Feature Extraction provide features to build model
        """
        self._random_state = seed
        pass

    def tfidf_vectorize(self, data, max_gram = 3, max_features = 50000):
        """
            Convert text data into TF-IDF features
        """
        tfid_vectorizer = TfidfVectorizer(ngram_range=(1,max_gram), max_features=max_features)

        return tfid_vectorizer.fit_transform(data)

    def hasing_vectorize(self, data, n_features = 500):
        """
            Convert text data into hasing vectorize features
        """
        vectorizer = HashingVectorizer(n_features=n_features)
        return vectorizer.transform(data)
    
    def categorical_to_dummy(self, data, column):
        """
            Convert categorical column into dummy columns
        """
        copy_data = data.copy()
        dummy_col = data[column].unique()[1:]
        for val in dummy_col:                                   # Use k-1 dummy columns for k classes
            copy_data[f"{column}_{val}"] = data[column] == val
            copy_data[f"{column}_{val}"] = copy_data[f"{column}_{val}"].astype(int)
        
        return copy_data.drop(columns=[column])

    def split_dataset(self, X, y, ratio=0.8):
        X_train, X_val, y_train, y_val = train_test_split(X, y, 
                                                                train_size=ratio, 
                                                                random_state=self._random_state,
                                                                stratify=y
                                                                )
        
        return X_train, X_val, y_train, y_val