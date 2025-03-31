package org.queueinc.hamlet.automl

import org.queueinc.hamlet.controller.*
import java.io.BufferedReader
import java.io.IOException
import java.io.InputStream
import java.io.InputStreamReader
import java.util.*
import java.util.stream.Stream

fun getOutputFromProgram(program: Array<String>) {
    val proc = Runtime.getRuntime().exec(program)

    println("Here is the standard output/error of the command:\n")

    Stream.of(proc.errorStream, proc.inputStream).parallel().forEach { isForOutput: InputStream ->
        try {
            BufferedReader(InputStreamReader(isForOutput)).use { br ->
                var line: String?
                while (br.readLine().also { line = it } != null) {
                    println(line)
                }
            }
        } catch (e: IOException) {
            throw RuntimeException(e)
        }
    }

    proc.waitFor()
    proc.destroy()
}

fun execAutoML(workspacePath: String, config: Config, debug: Boolean) {

    val version = Properties().let {
        it.load(Controller::class.java.getResourceAsStream("/version.properties"))
        it.getProperty("version")
    }

    val image = if (debug) "automl-image" else "ghcr.io/queueinc/automl-container:$version"

    val copy = { source: String, destination: String, target: String ->
        arrayOf("docker", "run", "-v", "$source:/source", "-v", "$destination:/dest", "-w", "/source", "alpine", "cp", "-r", target, "/dest")
    }

    val createVolume = arrayOf("docker", "volume", "create", "dummy_volume")
    val removeVolume = arrayOf("docker", "volume", "remove", "dummy_volume")
    val exec  =
        arrayOf("docker", "run", "--rm", "--volume", "dummy_volume:/test", image, "python", "automl/main.py",
                "--dataset", config.dataset,
                "--metric", config.metric,
                "--fair_metric", config.fairnessMetric,
                "--sensitive_features", config.sensitiveFeatures,
                "--mode", config.mode,
                "--batch_size", config.batchSize.toString(),
                "--time_budget", config.timeBudget.toString(),
                "--seed", config.seed.toString(),
                "--input_path", "/test/automl_input_${config.iteration}.json",
                "--output_path", "/test/automl_output_${config.iteration}.json")

    if (debug) {
        val build = arrayOf("docker", "build", "-t", "automl-container", ".")
        getOutputFromProgram(build)
    }

    getOutputFromProgram(createVolume)
    getOutputFromProgram(copy("$workspacePath/automl/input/", "dummy_volume", "automl_input_${config.iteration}.json"))
    getOutputFromProgram(exec)
    getOutputFromProgram(copy("dummy_volume", "$workspacePath/automl/output/", "automl_output_${config.iteration}.json"))
    getOutputFromProgram(copy("dummy_volume", "$workspacePath/automl/output/", "automl_output_${config.iteration}.csv"))
    getOutputFromProgram(removeVolume)

    println("AutoML execution ended")
}