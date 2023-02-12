import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin


class TextProcessor(BaseEstimator, TransformerMixin):

    def fit(self, x, y=None):
        return self

    def transform(self, x: pd.Series, y=None):
        return x.map(self.process_text)

    @staticmethod
    def process_text(text: str) -> str:
        text = text.replace('-', ' ')
        text = text.replace('#', '')
        words = []

        for word in text.split(' '):
            word = word.replace('-', ' ')
            word = ''.join(c for c in word if c.isalpha())

            if len(word) > 1:
                words.append(word)

        return ' '.join(words)
