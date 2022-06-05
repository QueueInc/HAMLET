package org.queueinc.hamlet.controller

import org.queueinc.hamlet.dictionary

typealias Point = List<String>

data class Rule(
    val type: String,
    private val rule: List<String>,
    val occurrences: Int,
    private val considered_configurations: Int,
    val support: Float,
    private val metric_threshold: Float) {

    val consideredConfigurations : Int
        get() = considered_configurations
    val metricThreshold : Float
        get() = metric_threshold

    val algorithm : String
        get() = dictionary.filter { rule.contains(it.value) }.map { it.key }.first()
    val steps  : List<String>
        get() = rule.filterNot { el -> dictionary.values.contains(el) }
    val theoryRepresentation : String
        get() = "$type([${ steps.joinToString(",") }], ${algorithm})"
    val isValid : Boolean
        get() = dictionary.count { rule.contains(it.value) } == 1

}

data class AutoMLResults(val evaluatedPoints: List<Point>, val inferredRules: List<Rule>)
