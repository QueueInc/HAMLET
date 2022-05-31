package org.queueinc.hamlet

import it.unibo.tuprolog.core.Term
import java.io.File

fun Term.toSklearnClass() =
    when(this.toString()) {
        "dt" -> "DecisionTreeClassifier"
        "knn" -> "KNeighborsClassifier"
        "naive_bayes" -> "GaussianNB"
        "function_transformer" -> "FunctionTransformer"
        "robust_scaler" -> "RobustScaler"
        "kbins" -> "KBinsDiscretizer"
        "binarizer" -> "Binarizer"
        "power_transformer" -> "PowerTransformer"
        "standard" -> "StandardScaler"
        "minmax" -> "MinMaxScaler"
        "select_k_best" -> "SelectKBest"
        "pca" -> "PCA"
        "simple_imputer" -> "SimpleImputer"
        "iterative_imputer" -> "IterativeImputer"
        else -> this.toString()
    }

fun File.createAndWrite(text: String) {
    this.createNewFile()
    this.writeText(text)
}