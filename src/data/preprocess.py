import pandas as pd
import nltk
import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from pathlib import Path
from nltk.tag import pos_tag

# Make sure that all resource of model has been downloaded
NLTK_RESOURCES = {
    'punkt': 'tokenizers/punkt',
    'stopwords': 'corpora/stopwords',
    'wordnet': 'corpora/wordnet',
    'averaged_perceptron_tagger': 'taggers/averaged_perceptron_tagger',
}

def ensure_nltk_resources(resources=None):
    """
    Check if NLTK resources are available, download them if not.
    
    :param resources: List of resource names to ensure (from NLTK_RESOURCES). If None, all will be checked.
    """
    resources = resources or NLTK_RESOURCES.keys()
    for name in resources:
        resource_path = NLTK_RESOURCES.get(name)
        if not resource_path:
            print(f"Unknown resource: {name}")
            continue
        try:
            nltk.data.find(resource_path)
            print(f"✓ '{name}' is already available.")
        except LookupError:
            print(f"✗ '{name}' not found. Downloading...")
            nltk.download(name)


class Preprocessing:
    def __init__(self):
        # Download resource
        ensure_nltk_resources()

    def read_CSV(self, file_name = 'test.csv'):
        # Get the absolute path of the current file
        current_dir = Path(__file__).resolve().parent

        # Navigate two levels up to the project root and access data/test.csv
        csv_path = current_dir.parent.parent / "data" / file_name
        return pd.read_csv(csv_path, encoding='unicode_escape').dropna()

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
    
    def preprocess(self, text, return_tokens=False):
        func_list = [self.pre_clean_text, self.pre_text_lowercase, 
                     self.pre_remove_punctuation, self.pre_tokenize,
                     self.pre_remove_stopwords, self.pre_lemmatize]
        
        try:
            for func in func_list:
                text = func(text)
        except:
            raise ValueError(f"Error at text: {text}")
        if return_tokens:
            # print(text)
            return text
        else:
            return " ".join(text)
    
    
    def tokenize_preprocessing(self, text, return_token=True):
        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        tokens = nltk.word_tokenize(text)
        
        # Remove stopword
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words]

        # Lemmatize
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(word) for word in tokens]

        # Remove number and particular name tokens
        pos_tags = pos_tag(tokens)
        processed_tokens = []
        for word, tag in pos_tags:
            if re.fullmatch(r'\d+', word):  # If token is a number
                continue
            elif tag in ['NNP', 'NNPS']:  # Proper nouns (singular/plural)
                continue
            processed_tokens.append(word)

        if not return_token:
            return ' '.join(processed_tokens)
        
        return processed_tokens
    
if __name__ == "__main__":
    pre_proc = Preprocessing()
    df = pre_proc.read_CSV('test.csv')
    print(df.info())