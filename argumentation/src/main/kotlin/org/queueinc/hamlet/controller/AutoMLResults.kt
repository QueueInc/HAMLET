package org.queueinc.hamlet.controller

import org.queueinc.hamlet.dictionary
import kotlin.random.Random

typealias Point = List<String>

data class Rule(
    val source: String,
    val type: String,
    val group: String,
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

    val isValid : Boolean
        get() = dictionary.count { rule.contains(it.value) } == 1

    val theoryRepresentation : String
        get() = if (type == "discriminate")
            "$type(pipeline([${ steps.joinToString(",") }], ${algorithm}), [${group.split("_").joinToString(",")}])"
        else "$type([${ steps.joinToString(",") }], ${algorithm})"

    val theory : String
        get() = if (type == "discriminate") "$theoryRepresentation." else "cc${Random.nextLong(0, Long.MAX_VALUE)} : [] => $theoryRepresentation. % $source"

}

data class AutoMLResults(val evaluatedPoints: List<Point>, val inferredRules: List<Rule>)
