import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin

from aif360.sklearn.preprocessing import LearnedFairRepresentations


class LFR_wrapper(BaseEstimator, TransformerMixin):
    """
    Wrapper class for LearnedFairRepresentations that allows it to be used with numpy arrays.
    """

    feature_names = []

    def __init__(self, prot_attr, feature_names, **kwargs):
        """
        Initialize the LFR_wrapper.

        Args:
            X: The numpy array of input data.
            y: The numpy array of target labels.
            feature_names: The list of feature names.
            **kwargs: Additional keyword arguments to pass to the LearnedFairRepresentations constructor.
        """
        self.prot_attr = prot_attr
        self.feature_names = feature_names
        self.lfr = LearnedFairRepresentations(prot_attr=prot_attr, **kwargs)

    def fit(self, X, y):
        df = pd.DataFrame(X, columns=self.feature_names).set_index(self.prot_attr)
        df["target"] = y
        self.lfr.fit(df.drop(columns=["target"]), df["target"])
        return self

    def transform(self, X):
        df = pd.DataFrame(X, columns=self.feature_names).set_index(self.prot_attr)
        transformed_df = self.lfr.transform(df)
        return transformed_df.values

    def fit_transform(self, X, y=None):
        df = pd.DataFrame(X, columns=self.feature_names).set_index(self.prot_attr)
        df["target"] = y
        transformed_df = self.lfr.fit_transform(
            df.drop(columns=["target"]), df["target"]
        )
        return transformed_df.values
