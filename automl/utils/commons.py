from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
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
    "SVC",
    "RandomForestClassifier",
    "AdaBoostClassifier",
    "ExtraTreesClassifier",
    "LogisticRegression",
    "XGBClassifier",
    "LGBMClassifier",
    "CatBoostClassifier"
]
