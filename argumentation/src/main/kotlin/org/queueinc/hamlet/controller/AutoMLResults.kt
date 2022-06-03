package org.queueinc.hamlet.controller

typealias Point = List<String>

data class Rule(val type: String, val rule: List<String>, val support: Float, val metric_threshold: Float) {
    fun theoryRepresentation() = "$type([${ rule.joinToString(",") }], ${rule[0]})"
}

data class AutoMLResults(val evaluatedPoints: List<Point>, val inferredRules: List<Rule>)
