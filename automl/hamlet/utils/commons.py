import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

algorithms = [
    "KNeighborsClassifier",
    "DecisionTreeClassifier",
    "GaussianNB",
    "MLPClassifier",
    "LinearSVC",
    "RandomForestClassifier",
    "AdaBoostClassifier",
    "ExtraTreesClassifier",
    "LogisticRegression",
    "XGBClassifier",
    "LGBMClassifier",
    "CatBoostClassifier",
]


def create_directory(result_path):

    if not os.path.exists(result_path):
        os.makedirs(result_path)

    return result_path
