from pgmpy.models import BayesianNetwork
from pgmpy.inference import VariableElimination
from pgmpy.factors.discrete import TabularCPD
from collections import Counter
from itertools import product
import nltk

class SentimentBayesianNetwork:
    def __init__(self, df, pre_proc):
        """
        :param df: A dataframe containing 2 columns ["text","sentiment"]
        :param pre_proc: A preprocessor to preprocessor the text columns
        """
        self.data, self.vocab = self.compute_word_frequencies(df, pre_proc)
        self.pre_proc = pre_proc
        self.labels = df["sentiment"].unique()  # ["neutral", "positive","negative"]
        self.label_prob = df["sentiment"].value_counts(normalize=True).tolist()
        self.graph = (list(product(["Sentiment"], self.vocab)))  # [("Sentiment","word1"), ("Sentiment","word2"),...]
        self.model = BayesianNetwork(self.graph[:len(self.vocab)]) # TODO: passing graph directly causes error?
        self._define_cpds(df)

    def _define_cpds(self, df):
        """
        Assuming each word is independent
        :param df: A dataframe containing 2 columns ["text","sentiment"]
        :return:None
        """
        # P(Sentiment)
        cpd_sentiment = TabularCPD(variable='Sentiment', variable_card=3,
                                   values=[[p] for p in self.label_prob]
                                   )
        self.model.add_cpds(cpd_sentiment)


        # P(Words|Sentiment)
        vocab_size = len(self.vocab)
        total_word_counts = {l: sum(self.data[l].values()) for l in self.labels}  # Per sentiment

        for word in self.vocab:
            word_counts = [self.data[l][word] for l in self.labels]
            total_counts = [total_word_counts[l] + vocab_size for l in self.labels]  # Add-V Smoothing

            prob_values = [[wc / tc] for wc, tc in zip(word_counts, total_counts)]

            cpd_word = TabularCPD(
                variable=word,
                variable_card=2,  # Word=0,1
                values=[
                    [1 - prob_values[i][0] for i in range(len(self.labels))],  # P(Word=0 | Sentiment)
                    [prob_values[i][0] for i in range(len(self.labels))]       # P(Word=1 | Sentiment)
                ],
                evidence=["Sentiment"],
                evidence_card=[len(self.labels)]  # ["neutral", "positive","negative"]
            )
            self.model.add_cpds(cpd_word)

        assert self.model.check_model()

    def infer(self, text_value: str):
        """
        Preprocesses the input text with the preprocessor,
        then performs sentiment inference using the Bayesian Network.

        text_value : str
            The input text for sentiment prediction.

        Returns:
        -------
            - result.values (list): A list of probabilities corresponding to each sentiment class.
            - self.labels (list): A list of sentiment labels (e.g., ['negative', 'neutral', 'positive']).
        """
        words = self.pre_proc.preprocess(text_value,True)  # Preprocess input text

        evidence = {word: 1 for word in words if word in self.vocab}

        inference = VariableElimination(self.model)
        result = inference.query(variables=['Sentiment'], evidence=evidence)

        return result.values, self.labels

    def display_network(self):
        """
        Display the edges in the bayesian network
        :return: None
        """
        return self.model.edges()

    def display_cpds_table(self):
        """
        Display all Conditional Probability Distributions tables.
        :return: None
        """
        for cpd in self.model.get_cpds():
            print(cpd)

    def compute_word_frequencies(self, df, pre_proc):
        """
        Compute word frequencies per sentiment, ensuring all words in the vocabulary are counted
        with a minimum count of 1

        :param df: DataFrame containing 'text' and 'sentiment' columns.
        :param pre_proc: Preprocessing object to clean and tokenize text.
        :return: Dictionary of n-Counter, one per sentiment category.
        """
        if 'text' not in df.columns or 'sentiment' not in df.columns:
            raise ValueError("Dataset must contain 'text' and 'sentiment' columns")

        # Process text column for entire dataset at once
        df["processed_text"] = df["text"].apply(lambda x: pre_proc.preprocess(x, return_tokens=True))
        

        vocabulary = set()
        
        for tokens in df['processed_text']:
            vocabulary.update(tokens)  # Add words to the vocabulary set, note: update expect an iterable not a literal

        print("Vocavulary size: ", len(vocabulary))
        # Create a Counter for each sentiment also add-1
        sentiment_word_counts = {sentiment: Counter() for sentiment in df['sentiment'].unique()}
        for sentiment in df['sentiment'].unique():
            sentiment_word_counts[sentiment].update(vocabulary)

        # Update each sentiment counter
        for sentiment, group in df.groupby('sentiment'):
            for tokens in group['processed_text']:
                sentiment_word_counts[sentiment].update(tokens)
        
        return sentiment_word_counts, vocabulary

    def view_data(self):
        return self.data


