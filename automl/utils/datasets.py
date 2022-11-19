# OpenML provides several benchmark datasets
import openml
import os

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


def load_from_csv(id, input_path=os.path.join("/home", "resources", "datasets")):
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
