"""Class Columns used to apply features transformation."""
from sklearn.base import BaseEstimator, TransformerMixin


class Columns(BaseEstimator, TransformerMixin):
    """Extract names columns from dataframe."""

    def __init__(self, names=None):
        """Initialise a Columns instance.

        Parameters
        ----------
        names : list
            List of feature names to apply the transform on.

        """
        self.names = names

    def fit(self, X, y=None, **fit_params):
        """Fit to training set.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Training set.
        y : numpy array of shape [n_samples]
            Target values.
        **fit_params : type
            Description of parameter `**fit_params`.

        Returns
        -------
        self
            Returns an instance of self.

        """
        return self

    def transform(self, X):
        """Extract specific features prior transformation.

        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]
            The data used to transform along the features axis.

        Returns
        -------
        array-like, shape [n_samples, n_features]
            Transformed features.

        """
        # Select only columns present in dataset.
        filtered_names = list(filter(lambda x: x in self.names, X.columns))
        return X[filtered_names]
