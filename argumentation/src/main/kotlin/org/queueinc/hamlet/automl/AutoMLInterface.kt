package org.queueinc.hamlet.automl

import org.queueinc.hamlet.controller.*
import java.io.BufferedReader
import java.io.IOException
import java.io.InputStream
import java.io.InputStreamReader
import java.util.*
import java.util.stream.Stream
import kotlin.random.Random

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

    val r = Random.nextInt(0, Int.MAX_VALUE)

    val version = Properties().let {
        it.load(Controller::class.java.getResourceAsStream("/version.properties"))
        it.getProperty("version")
    }

    val image = if (debug) "automl-image" else "ghcr.io/queueinc/automl-container:$version"

    val create =
        arrayOf("docker", "container", "create", "--name", "hamlet_$r", image, "python", "automl/main.py",
            "--seed", config.seed.toString(),
            "--input_path", "/home/automl_input_${config.iteration}.json",
            "--output_path", "/home/automl_output_${config.iteration}.json")

    val copy = { source: String, destination: String ->
        arrayOf("docker", "cp", source, destination)
    }

    val exec  =
        arrayOf("docker", "start", "hamlet_$r", "-a")

    if (debug) {
        val build = arrayOf("docker", "build", "-t", "automl-image", "../")
        getOutputFromProgram(build)
    }

    getOutputFromProgram(create)
    getOutputFromProgram(copy("$workspacePath/automl/input/automl_input_${config.iteration}.json", "hamlet_$r:/home/automl_input_${config.iteration}.json"))
    getOutputFromProgram(exec)
    getOutputFromProgram(copy("hamlet_$r:/home/automl_output_${config.iteration}.json", "$workspacePath/automl/output/automl_output_${config.iteration}.json"))
    getOutputFromProgram(copy("hamlet_$r:/home/automl_output_${config.iteration}.csv", "$workspacePath/automl/output/automl_output_${config.iteration}.csv"))
    getOutputFromProgram(arrayOf("docker", "rm", "hamlet_$r"))

    println("AutoML execution ended")
}