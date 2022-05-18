package org.queueinc.hamlet

import it.unibo.tuprolog.core.Term

fun Term.toSklearnClass() =
    when(this.toString()) {
        "dt" -> "DecisionTreeClassifier"
        "knn" -> "KNeighborsClassifier"
        "kbins" -> "KBinsDiscretizer"
        "standard" -> "StandardScaler"
        "minmax" -> "MinMaxScaler"
        "functionTransformer" -> "FunctionTransformer"
        else -> this.toString()
    }