import subprocess
import openml
import math
import os
import argparse

import autosklearn.classification

import pandas as pd

from tqdm import tqdm
from sklearn.model_selection import cross_validate
from sklearn.model_selection import StratifiedKFold
from smac.optimizer.smbo import SMBO
from smac.runhistory.runhistory import RunInfo, RunValue
from classifiers.MyAdaboostClassifier import MyAdaboostClassifier
from classifiers.MyDecisionTreeClassifier import (
    MyDecisionTreeClassifier,
)
from classifiers.MyGaussianNaiveBayesClassifier import (
    MyGaussianNaiveBayesClassifier,
)

from classifiers.MyKNearestNeighborsClassifier import (
    MyKNearestNeighborsClassifier,
)
from classifiers.MyMultiLayerPerceptronClassifier import (
    MyMultiLayerPerceptronClassifier,
)
from classifiers.MyRandomForestClassifier import (
    MyRandomForestClassifier,
)
from classifiers.MySupportVectorClassifier import (
    MySupportVectorClassifier,
)

i = 0


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
    args = parser.parse_args()
    return args


def callback(smbo: SMBO, run_info: RunInfo, result: RunValue, time_left: float) -> bool:
    global i
    i += 1
    return i <= 1000


def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

    return directory


args = parse_args()
dataset = openml.datasets.get_dataset(args.id)
path = create_directory(os.path.join("resources", "auto-sklearn", dataset.name))

print(dataset.name)
print()

X, y, _, _ = dataset.get_data(
    dataset_format="array", target=dataset.default_target_attribute
)

cls = autosklearn.classification.AutoSklearnClassifier(
    time_left_for_this_task=7200,
    resampling_strategy=StratifiedKFold(n_splits=10),
    metric=autosklearn.metrics.balanced_accuracy,
    get_trials_callback=callback,
    seed=42,
)

cls.fit(X.copy(), y.copy(), dataset_name=dataset.name)

try:
    score = round(
        pd.DataFrame(cls.cv_results_)
        .sort_values("rank_test_scores")
        .iloc[0]["mean_test_score"]
        * 100,
        2,
    )
    print(score)
except Exception as e:
    print(e)

try:
    pd.DataFrame(cls.show_models()).T.to_csv(os.path.join(path, "models_details.csv"))
except Exception as e:
    print(e)
    
try:
    pd.DataFrame(cls.cv_results_).to_csv(os.path.join(path, "cv_results.csv"))
except Exception as e:
    print(e)
    
try:
    pd.DataFrame(cls.performance_over_time_).to_csv(
        os.path.join(path, "performance_over_time.csv")
    )
except Exception as e:
    print(e)
    
try:
    print(cls.sprint_statistics())
except Exception as e:
    print(e)
    
