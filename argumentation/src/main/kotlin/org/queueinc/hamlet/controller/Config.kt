package org.queueinc.hamlet.controller

data class Config(
    var iteration : Int,
    val dataset: String,
    val metric: String,
    val mode: String,
    val batchSize: Int,
    val seed: Int
)
