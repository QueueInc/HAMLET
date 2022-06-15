# OpenML provides several benchmark datasets
import openml
import pandas as pd
import numpy as np


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
    return load_dataset_from_openml(id)


def load_dataset_from_openml(id):
    dataset = openml.datasets.get_dataset(id)
    X, y, categorical_indicator, _ = dataset.get_data(
        dataset_format="array", target=dataset.default_target_attribute
    )
    # cat_features = [i for i, x in enumerate(categorical_indicator) if x == True]
    # Xt = pd.DataFrame(X)
    # Xt[cat_features] = Xt[cat_features].fillna(-1)
    # Xt[cat_features] = Xt[cat_features].astype("str")
    # Xt[cat_features] = Xt[cat_features].replace("-1", np.nan)
    # Xt = Xt.to_numpy()
    # return Xt, y, categorical_indicator
    return X, y, categorical_indicator
