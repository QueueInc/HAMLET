# OpenML provides several benchmark datasets
import json
import openml
import os
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer


def get_dataset_by_name(name):
    loader = {
        "blood": 1464,
        "breast-t": 1465,  # this is breast-tissue, not breast cancer
        "breast-w": 15,
        "diabetes": 37,
        "ecoli": 40671,
        "iris": 61,
        "parkinsons": 1488,
        "seeds": 1499,
        "thyroid": 40682,
        "vehicle": 54,
        "wine": 187,
    }
    if name in loader:
        return load_dataset_from_openml(loader[name])
    else:
        raise Exception("There is no such a dataset in the loader")


def get_dataset_by_id(id):
    print(__file__)
    return load_dataset_from_openml(id)


def load_dataset_from_openml(
    id,
    sensitive_features,
    input_path=os.path.join(
        Path(__file__).parent.parent.parent.resolve(), "resources", "datasets"
    ),
):
    dataset = openml.datasets.get_dataset(id)
    X, y, categorical_indicator, feature_names = dataset.get_data(
        dataset_format="array", target=dataset.default_target_attribute
    )
    # with open(os.path.join(input_path, "sensitive_indicators.json")) as f:
    #     sensitive_indicators = json.load(f)
    # sensitive_indicator = sensitive_indicators[str(id)]
    sensitive_indicator = [
        True if x in [int(y) for y in sensitive_features.split("_")] else False
        for x in range(len(categorical_indicator))
    ]

    if id == "179":
        X_temp = np.concatenate([X, y.reshape(-1, 1)], axis=1)
        X_temp = X_temp[~np.isnan(X_temp).any(axis=1)]
        X, y = X_temp[:, :-1], X_temp[:, -1].T
    if id == "31":
        est = KBinsDiscretizer(
            n_bins=5, encode="ordinal", strategy="kmeans"
        )  # strategy{"uniform", "quantile", "kmeans"}
        X[:, 12] = est.fit_transform(X[:, 12].reshape(-1, 1)).ravel()
        categorical_indicator[12] = True
    # cat_features = [i for i, x in enumerate(categorical_indicator) if x == True]
    # Xt = pd.DataFrame(X)
    # Xt[cat_features] = Xt[cat_features].fillna(-1)
    # Xt[cat_features] = Xt[cat_features].astype("str")
    # Xt[cat_features] = Xt[cat_features].replace("-1", np.nan)
    # Xt = Xt.to_numpy()
    # return Xt, y, categorical_indicator
    return X, y, categorical_indicator, sensitive_indicator, feature_names


def load_from_csv(
    id,
    input_path=os.path.join(
        Path(__file__).parent.parent.parent.resolve(), "resources", "datasets"
    ),
):
    """Load a dataset given its id on OpenML from resources/datasets.

    Args:
        id: id of the dataset.

    Returns:
        numpy.array: data items (X) of the dataset.
        numpy.array: target (y) of the dataset.
        list: mask that indicates categorical features.
    """
    import pandas as pd
    import json

    df = pd.read_csv(os.path.join(input_path, f"{id}.csv"))
    with open(os.path.join(input_path, "categorical_indicators.json")) as f:
        categorical_indicators = json.load(f)
    categorical_indicator = categorical_indicators[str(id)]
    X, y = df.iloc[:, :-1].to_numpy(), df.iloc[:, -1].to_numpy()
    return X, y, categorical_indicator
