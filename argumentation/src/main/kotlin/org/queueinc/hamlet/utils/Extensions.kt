package org.queueinc.hamlet

import it.unibo.tuprolog.core.Term
import java.io.File

val dictionary = mapOf(
    "dt" to "DecisionTreeClassifier",
    "xgboost" to "XGBClassifier",
    "lightboost" to "LGBMClassifier",
    "catboost" to "CatBoostClassifier",
    "knn" to "KNeighborsClassifier",
    "naive_bayes" to "GaussianNB",
    "nn" to "MLPClassifier",
    "svm" to "SVC",
    "rf" to "RandomForestClassifier",
    "ada_boost" to "AdaBoostClassifier",
    "function_transformer" to "FunctionTransformer",
    "robust_scaler" to "RobustScaler",
    "kbins" to "KBinsDiscretizer",
    "binarizer" to "Binarizer",
    "power_transformer" to "PowerTransformer",
    "standard" to "StandardScaler",
    "minmax" to "MinMaxScaler",
    "select_k_best" to "SelectKBest",
    "pca" to "PCA",
    "simple_imputer" to "SimpleImputer",
    "iterative_imputer" to "IterativeImputer",
    "ordinal_encoder" to "OrdinalEncoder",
    "smote" to "SMOTE",
    "near_miss" to "NearMiss"

)

fun Term.toSklearnClass() = dictionary[this.toString()] ?: this.toString()

fun File.createAndWrite(text: String) {
    this.createNewFile()
    this.writeText(text)
}
