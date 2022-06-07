package org.queueinc.hamlet.automl

import org.queueinc.hamlet.controller.*
import java.io.BufferedReader
import java.io.InputStreamReader
import java.util.*


fun stopAutoML() {
    val stop = arrayOf("docker", "stop", "automl-container")
    val rm = arrayOf("docker", "rm", "automl-container")

    Runtime.getRuntime().exec(stop).waitFor()
    Runtime.getRuntime().exec(rm).waitFor()
}

fun runAutoML(workspacePath: String, debug: Boolean) {

    if (debug) {
        val build = arrayOf("docker", "build", "-t", "automl-container", ".")
        val run = arrayOf("docker", "run", "--name", "automl-container", "--volume", "${System.getProperty("user.dir")}/automl:/home/automl",
            "--volume", "${workspacePath}:/home/resources", "--detach", "-t", "automl-container")

        Runtime.getRuntime().exec(build).waitFor()
        Runtime.getRuntime().exec(run).waitFor()

        return
    }

    val version = Properties().let {
        it.load(Controller::class.java.getResourceAsStream("/version.properties"))
        it.getProperty("version")
    }

    val run = arrayOf("docker", "run", "--name", "automl-container",
        "--volume", "${workspacePath}:/home/resources", "--detach", "-t", "ghcr.io/queueinc/automl-container:$version")

    Runtime.getRuntime().exec(run).waitFor()
}

fun execAutoML(iteration: Int) {

    val exec  =
        arrayOf("docker", "exec", "automl-container", "python", "automl/main.py",
                "--dataset", dataset, "--metric", metric, "--mode", mode, "--batch_size", batchSize.toString(), "--seed", seed.toString(),
                "--input_path", "/home/resources/automl/input/automl_input_${iteration}.json",
                "--output_path", "/home/resources/automl/output/automl_output_${iteration}.json")

    val proc = Runtime.getRuntime().exec(exec)
    val stdInput = BufferedReader(InputStreamReader(proc.inputStream))
    val stdError = BufferedReader(InputStreamReader(proc.errorStream))

    println("Here is the standard output/error of the command:\n")
    var s: String?
    while (stdError.readLine().also { s = it } != null || stdInput.readLine().also { s = it } != null) {
        println(s)
    }

    println("AutoML execution ended")
}