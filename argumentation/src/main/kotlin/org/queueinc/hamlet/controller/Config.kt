package org.queueinc.hamlet.controller

data class Config(
    var iteration : Int,
    val dataset: String?,
    val metric: String?,
    val fairnessMetric: String?,
    val sensitiveFeatures: List<String>,
    val mode: String,
    val batchSize: Int,
    val timeBudget: Int,
    val seed: Int,
    val fairnessThresholds: Pair<Double, Double> = Pair(0.4, 0.6),
    val performanceThresholds: Pair<Double, Double> = Pair(0.4, 0.6),
    val miningSupport: Double = 0.6
)
