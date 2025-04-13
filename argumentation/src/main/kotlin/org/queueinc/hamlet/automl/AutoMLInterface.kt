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

    val copy = { source: String, destination: String ->
        listOf(
            arrayOf("docker", "container", "create", "--name", "dummy_$r", "-v", "dummy_volume_$r:/data", "tianon/true"),
            arrayOf("docker", "cp", source, destination),
            arrayOf("docker", "rm", "dummy_$r")
            //arrayOf("docker", "run", "--rm", "-v", "$source:/source", "-v", "$destination:/dest", "alpine", "cp", "-r", "/source/$target", "/dest/$target")
        )
    }


    //val createVolume = arrayOf("docker", "volume", "create", "dummy_volume_$r")
    //val removeVolume = arrayOf("docker", "volume", "remove", "dummy_volume_$r")
    val exec  =
        arrayOf("docker", "run", "--rm",
                //"--volume", "dummy_volume_$r:/data",
                "--volume", "$workspacePath/automl/input:/input",
                "--volume", "$workspacePath/automl/output:/output",
                image, "python", "automl/main.py",
                "--seed", config.seed.toString(),
                "--input_path", "/input/automl_input_${config.iteration}.json",
                "--output_path", "/output/automl_output_${config.iteration}.json")

    if (debug) {
        val build = arrayOf("docker", "build", "-t", "automl-image", "../")
        getOutputFromProgram(build)
    }

    //getOutputFromProgram(createVolume)
    //copy("$workspacePath/automl/input/automl_input_${config.iteration}.json", "dummy_volume_$r:/data/automl_input_${config.iteration}.json")
    //    .forEach{getOutputFromProgram(it)}
    getOutputFromProgram(exec)
    //copy("dummy_volume_$r:/data/automl_output_${config.iteration}.json", "$workspacePath/automl/output/")
    //    .forEach{getOutputFromProgram(it)}
    //copy("dummy_volume_$r:/data/automl_output_${config.iteration}.csv", "$workspacePath/automl/output/")
    //    .forEach{getOutputFromProgram(it)}
    //getOutputFromProgram(removeVolume)

    println("AutoML execution ended")
}