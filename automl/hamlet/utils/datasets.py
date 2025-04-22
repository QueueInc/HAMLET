# OpenML provides several benchmark datasets
import json
import openml
import os
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer, OrdinalEncoder


def get_dataset_by_name(name):
    loader = {
        "adult": 179,
        "blood": 1464,
        "breast-t": 1465,  # this is breast-tissue, not breast cancer
        "breast-w": 15,
        "compass": 44162,
        "credit-g": 31,
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
        return loader[name]
    else:
        raise Exception("There is no such a dataset in the loader")


def get_dataset_by_id(id):
    print(__file__)
    return load_dataset_from_openml(id)


def preprocess_features(df, sensitive_features, enc_categorical_features):
    """
    Discretizes non-categorical sensitive features and applies ordinal encoding to categorical features.

    Parameters:
    - df: pd.DataFrame
        The input DataFrame containing the data.
    - sensitive_features: list of str
        List of column names to be checked and discretized if non-categorical.

    Returns:
    - df_transformed: pd.DataFrame
        DataFrame with transformed features.
    - feature_mappings: dict
        Dictionary containing mappings for discretized and encoded features.
    """
    df_transformed = df.copy()

    categorical_features = (
        df.select_dtypes(include=["object", "category"]).columns.tolist()
        + enc_categorical_features
    )
    numerical_sensitive_features = [
        col
        for col in df.select_dtypes(exclude=["object", "category"]).columns.tolist()
        if col in sensitive_features and col not in enc_categorical_features
    ]

    # Encode all categorical features
    encoder = OrdinalEncoder()
    df_transformed[categorical_features] = encoder.fit_transform(
        df_transformed[categorical_features]
    )
    encoding_mappings = {
        feature: dict(zip(range(len(encoder.categories_[i])), encoder.categories_[i]))
        for i, feature in enumerate(categorical_features)
        if feature in sensitive_features
    }

    # Discretize non-categorical sensitive features
    for feature in numerical_sensitive_features:
        discretizer = KBinsDiscretizer(n_bins=5, encode="ordinal", strategy="kmeans")
        df_transformed[feature] = discretizer.fit_transform(
            df_transformed[[feature]]
        ).astype(int)
        bin_edges = discretizer.bin_edges_[0]
        bin_labels = [
            f"{bin_edges[i]:.2f} - {bin_edges[i+1]:.2f}"
            for i in range(len(bin_edges) - 1)
        ]
        encoding_mappings[feature] = dict(zip(range(len(bin_labels)), bin_labels))

    return df_transformed, encoding_mappings


def load_dataset_from_openml(
    name,
    sensitive_features,
    input_path=os.path.join(
        Path(__file__).parent.parent.parent.resolve(), "resources", "datasets"
    ),
):
    id = get_dataset_by_name(name)

    try:
        dataset = openml.datasets.get_dataset(id)
        df, _, categorical_indicator, feature_names = dataset.get_data(
            dataset_format="dataframe",
            # target=dataset.default_target_attribute
        )
        default_target_attribute = dataset.default_target_attribute
    except:
        df, categorical_indicator = load_from_csv(id)
        feature_names = list(df.columns)
        default_target_attribute = feature_names[-1]

    if id == 31:
        # Split the 'personal_status' into two new columns
        df[["sex", "personal_status"]] = df["personal_status"].str.split(
            " ", expand=True
        )
        df = df[["sex"] + feature_names]
        feature_names = list(df.columns)
        categorical_indicator = [True] + categorical_indicator

    group_percentages = df[sensitive_features].value_counts(normalize=True) * 100
    groups_to_drop = group_percentages[group_percentages < 1].index
    filtered_df = df[~df[sensitive_features].apply(tuple, axis=1).isin(groups_to_drop)]
    df = filtered_df.reset_index(drop=True)

    if default_target_attribute in feature_names and len(feature_names) == len(
        categorical_indicator
    ):
        categorical_indicator.pop(feature_names.index(default_target_attribute))
    feature_names = [col for col in feature_names if col != default_target_attribute]

    # Encode categorical and discretize numerical while storing the mapping
    df_transformed, encoding_mappings = preprocess_features(
        df,
        sensitive_features,
        [
            feature
            for idx, feature in enumerate(feature_names)
            if categorical_indicator[idx]
        ],
    )

    # Get old data structure
    X = df_transformed.drop(labels=default_target_attribute, axis="columns").to_numpy()
    y = df_transformed[default_target_attribute].to_numpy()

    sensitive_indicator = [feature in sensitive_features for feature in feature_names]
    encoding_mappings = {
        feature_names.index(key): value for key, value in encoding_mappings.items()
    }

    # with open(os.path.join(input_path, "sensitive_indicators.json")) as f:
    #     sensitive_indicators = json.load(f)
    # sensitive_indicator = sensitive_indicators[str(id)]

    # sensitive_indicator = [
    #     True if x in [int(y) for y in sensitive_features.split("_")] else False
    #     for x in range(len(categorical_indicator))
    # ]

    if id == 179:
        X_temp = np.concatenate([X, y.reshape(-1, 1)], axis=1)
        X_temp = X_temp[~np.isnan(X_temp).any(axis=1)]
        X, y = X_temp[:, :-1], X_temp[:, -1].T
    # cat_features = [i for i, x in enumerate(categorical_indicator) if x == True]
    # Xt = pd.DataFrame(X)
    # Xt[cat_features] = Xt[cat_features].fillna(-1)
    # Xt[cat_features] = Xt[cat_features].astype("str")
    # Xt[cat_features] = Xt[cat_features].replace("-1", np.nan)
    # Xt = Xt.to_numpy()
    # return Xt, y, categorical_indicator
    return (
        X,
        y,
        categorical_indicator,
        sensitive_indicator,
        feature_names,
        encoding_mappings,
    )


def load_from_csv(
    id,
    input_path=os.path.join(
        Path(__file__).parent.parent.parent.resolve(), "resources", "datasets"
    ),
):
    """Load a dataset given its id on OpenML from resources/datasets.

    All datasets in the folder are already encoded with an OrdinalEncoder fro mscikit-learn except datasets 31, 179, and 44162.
    Those datasets are also the only ones that have features names in the csv.

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
    return df, categorical_indicator


def load_from_csv_to_numpy(
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
    df, categorical_indicator = load_from_csv(id, input_path=input_path)
    X, y = df.iloc[:, :-1].to_numpy(), df.iloc[:, -1].to_numpy()
    return X, y, categorical_indicator
