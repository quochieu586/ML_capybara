from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

class FeatureExtraction:
    def __init__(self, seed = 2025):
        self._random_state = seed
        pass

    def tfidf_vectorize(self, data, max_gram = 3, max_features = 50000):
        tfid_vectorizer = TfidfVectorizer(ngram_range=(1,max_gram), max_features=max_features)

        return tfid_vectorizer.fit_transform(data)
    
    def split_dataset(self, X, y, train_size=0.8, validation_test_size=0.5):
        X_train, X_test_val, y_train, y_test_val = train_test_split(X, y, 
                                                                train_size=train_size, 
                                                                random_state=self._random_state,
                                                                stratify=y
                                                                )
        
        X_val, X_test, y_val, y_test  = train_test_split(X_test_val, y_test_val,
                                                        train_size=validation_test_size,
                                                        random_state=self._random_state,
                                                        stratify=y_test_val
                                                    )
        
        return X_train, X_val, X_test, y_train, y_val, y_test
