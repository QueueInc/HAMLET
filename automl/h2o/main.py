import os
import argparse

import openml

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score

import h2o
from h2o.automl import H2OAutoML


def parse_args():
    parser = argparse.ArgumentParser(
        description="Automated Machine Learning Workflow creation and configuration"
    )
    parser.add_argument(
        "-id",
        "--id",
        nargs="?",
        type=str,
        required=False,
    )
    parser.add_argument(
        "-budget",
        "--budget",
        nargs="?",
        type=int,
        required=False,
    )
    parser.add_argument(
        "-path",
        "--path",
        nargs="?",
        type=str,
        required=False,
    )
    args = parser.parse_args()
    return args


def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

    return directory


args = parse_args()

# Start the H2O cluster (locally)
h2o.init()

dataset = openml.datasets.get_dataset(args.id)
path = create_directory(os.path.join(args.path, args.id))

# Load dataset
print(dataset.name)
X, y, categorical_indicator, attribute_names = dataset.get_data(
    dataset_format="dataframe", target=dataset.default_target_attribute
)
df = pd.concat([X, y], axis=1)

# Create fold column
skf = StratifiedKFold(n_splits=10)
fold_map = {
    index: fold
    for fold, (_, test_index) in enumerate(skf.split(X, y))
    for index in test_index
}
df["fold"] = df.index.map(lambda row: fold_map[row])

# Convert to h2o frame
data = h2o.H2OFrame(df)

# For binary classification, response should be a factor
# if data[y.name].unique().shape[0] == 2:
data[y.name] = data[y.name].asfactor()
print(data)

# Instantiate AutoML Classifier
aml = H2OAutoML(
    keep_cross_validation_predictions=True,
    max_runtime_secs=args.budget,
    max_models=1000 if args.budget == 7200 else 500,
    seed=42,
    exclude_algos = ["StackedEnsemble"]
)

# Train AutoML Classifier
aml.train(x=attribute_names, y=y.name, training_frame=data, fold_column="fold")

try:
    # View the AutoML Leaderboard
    lb = aml.leaderboard
    print(lb.head(rows=lb.nrows))
except Exception as e:
    print(e)

try:
    # Calculate balanced accuracy
    cv_results = aml.leader.cross_validation_holdout_predictions().as_data_frame()
    cv_results.to_csv(os.path.join(path, "raw_cv_results.csv"))
    cv_results["class"] = df[y.name]
    cv_results.to_csv(os.path.join(path, "raw_cv_results.csv"))
    cv_results["fold"] = df.index.map(lambda row: fold_map[row])
    cv_results.to_csv(os.path.join(path, "raw_cv_results.csv"))
    print(
        cv_results.groupby(by="fold")
        .apply(lambda x: balanced_accuracy_score(x["class"].astype(int), x["predict"]))
        .mean()
    )
except Exception as e:
    print(e)
