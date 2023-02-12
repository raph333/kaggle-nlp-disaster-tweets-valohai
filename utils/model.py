from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.pipeline import Pipeline

from utils.data_processing import TextProcessor

MODEL_PIPELINE = Pipeline([
    ('text_processer', TextProcessor()),
    ('vectorizer', TfidfVectorizer(max_df=0.8,
                                   max_features=100000,
                                   ngram_range=(1, 3),
                                   stop_words='english',
                                   lowercase=True)
     ),
    ('model', LogisticRegressionCV(random_state=42,
                                   n_jobs=-1,
                                   max_iter=10000)
     )
])
