
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd 

class missing_transformer(BaseEstimator, TransformerMixin):

    def fit(self, X, y = None):
        return self

    def transform(self, X, y = None):
        return X.isna().any(axis = 1).to_frame(name = "na_ind")

    def fit_transform(self, X, y = None):
        return self.transform(X)