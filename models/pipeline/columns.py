"""Class Columns used to apply features transformation."""
from sklearn.base import BaseEstimator, TransformerMixin


class Columns(BaseEstimator, TransformerMixin):
    """Extract names columns from dataframe."""

    def __init__(self, names=None):
        self.names = names

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X):
        """Extract specific DataFrame columns during transform."""
        # Select only columns present in dataset.
        filtered_names = list(filter(lambda x: x in self.names, X.columns))
        return X[filtered_names]
