import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OrdinalEncoder

from aif360.sklearn.preprocessing import LearnedFairRepresentations


class LFR_wrapper(BaseEstimator, TransformerMixin):
    """
    Wrapper class for LearnedFairRepresentations that allows it to be used with numpy arrays.
    """

    feature_names = []
    enc = None

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
        self.lfr = LearnedFairRepresentations(
            prot_attr=prot_attr if len(prot_attr) == 1 else "mixin", **kwargs
        )

    def _prepare_dataset(self, X, y=None, fit=False):
        df = pd.DataFrame(X, columns=self.feature_names)

        if len(self.prot_attr) == 0:
            raise Exception("No sensitive features to mitigate")

        if len(self.prot_attr) > 1:
            df["mixin"] = df[self.prot_attr[0]].astype(str) + df[
                self.prot_attr[1]
            ].astype(str)
            df = df.drop(columns=self.prot_attr)

            mixin_numpy = df["mixin"].to_numpy().reshape(-1, 1)
            if fit:
                self.enc = OrdinalEncoder(
                    handle_unknown="use_encoded_value", unknown_value=-1
                )
                self.enc.fit(mixin_numpy)

            df["mixin"] = self.enc.transform(mixin_numpy)
            df = df.set_index("mixin")
        else:
            df = df.set_index(self.prot_attr)

        if y is not None:
            df["target"] = y
            df_X, df_y = df.drop(columns=["target"]), df["target"]
        else:
            df_X, df_y = df, None

        return df_X, df_y

    def fit(self, X, y):
        df_X, df_y = self._prepare_dataset(X, y, fit=True)
        self.lfr.fit(df_X, df_y)
        return self

    def transform(self, X):
        df_X, _ = self._prepare_dataset(X)
        transformed_df = self.lfr.transform(df_X)
        return transformed_df.values

    def fit_transform(self, X, y=None):
        df_X, df_y = self._prepare_dataset(X, y, fit=True)
        transformed_df = self.lfr.fit_transform(df_X, df_y)
        return transformed_df.values
