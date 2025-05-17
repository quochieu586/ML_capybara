import os
from pathlib import Path
import numpy as np

import gensim.downloader as api
from gensim.models import KeyedVectors

PRETRAINED_MODEL = 'fasttext-wiki-news-subwords-300'

class WordEmbeddingVectors:
    def __init__(self):
        current_dir = Path(__file__).resolve().parent
        # Navigate two levels up to the project root and access data/test.csv
        path = current_dir.parent.parent.parent / "data"

        for dirname, _, filenames in os.walk(path):
            for filename in filenames:
                print(os.path.join(dirname, filename))
        model_file = path / "fasttext_subwords_300.kv"

        if not model_file.exists():
            print("Not found model, install the new one !")
            self.model = api.load(PRETRAINED_MODEL)  # ~1GB, loads in RAM
            self.model.save(str(model_file))
        else:
            self.model = KeyedVectors.load(str(model_file), mmap='r')

        print(f"Loader model successfully ! Vector size: {self.model.vector_size}")

    def vectorize_sentence(self, sentence):
        vectors = [self.model[word] for word in sentence if word in self.model]
        if vectors:
            return np.mean(vectors, axis=0)
        else:
            return np.zeros(self.model.vector_size)
    
    def vectorize(self, tokens):
        return np.array([self.vectorize_sentence(s) for s in tokens])