import pandas as pd
import nltk
import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from pathlib import Path

class Preprocessing:
    def read_CSV(self, file_name):
        # Get the absolute path of the current file
        current_dir = Path(__file__).resolve().parent

        # Navigate two levels up to the project root and access data/test.csv
        csv_path = current_dir.parent.parent / "data" / file_name
        return pd.read_csv(csv_path).dropna()

    def pre_clean_text(self, text):
        return re.sub(r'[^A-Za-z(),!?\'\` ]', '', text)
    
    def pre_text_lowercase(self, text):
        """
        convert all words to lower case
        """
        return text.lower()
    
    def pre_remove_punctuation(self, text):
        """
        remove punctuation from text
        """
        return re.sub(f'[{string.punctuation}]', '', text)
    
    def pre_tokenize(self, text):
        """
        tokenize the text after cleaning
        """
        return nltk.word_tokenize(text)
    
    def pre_remove_stopwords(self, tokens):
        """
        remove stopwords from list of tokens
        """
        if not hasattr(self, "_stop_words"):
            self._stop_words = set(stopwords.words('english'))

        return [token for token in tokens if token not in self._stop_words]
    
    def pre_lemmatize(self, tokens):
        """
        lemmatize list of tokens
        """
        if not hasattr(self, "_lemmatizer"):
            self._lemmatizer = WordNetLemmatizer()
        
        return [self._lemmatizer.lemmatize(token) for token in tokens]
    
    def preprocess(self, text):
        func_list = [self.pre_clean_text, self.pre_text_lowercase, 
                     self.pre_remove_punctuation, self.pre_tokenize,
                     self.pre_remove_stopwords, self.pre_lemmatize]
        
        try:
            for func in func_list:
                text = func(text)
        except:
            raise ValueError(f"Error at text: {text}")

        return text
    
if __name__ == "__main__":
    pre_proc = Preprocessing()
    df = pre_proc.read_CSV('test.csv')
    print(df.info())
    # print(pre_proc.preprocess())
