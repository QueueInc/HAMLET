package org.queueinc.hamlet.automl

import org.queueinc.hamlet.controller.*
import java.io.BufferedReader
import java.io.InputStreamReader

fun stopAutoML() {
    val stop = arrayOf("docker", "stop", "automl_container")
    val rm = arrayOf("docker", "rm", "automl_container")

    Runtime.getRuntime().exec(stop).waitFor()
    Runtime.getRuntime().exec(rm).waitFor()
}

fun runAutoML(workspacePath: String) {

    val build = arrayOf("docker", "build", "-t", "automl_container", ".")
    val run = arrayOf("docker", "run", "--name", "automl_container", "--volume", "${System.getProperty("user.dir")}/automl:/home/automl",
        "--volume", "${workspacePath}:/home/resources", "--detach", "-t", "automl_container")

    Runtime.getRuntime().exec(build).waitFor()
    Runtime.getRuntime().exec(run).waitFor()
}

fun execAutoML(iteration: Int, dockerMode: Boolean) {

    val exec  = if (dockerMode)
        arrayOf("docker", "exec", "automl_container", "python", "automl/main.py",
                "--dataset", dataset, "--metric", metric, "--mode", mode, "--batch_size", batchSize.toString(), "--seed", seed.toString(),
                "--input_path", "/home/resources/automl/input/automl_input_${iteration}.json",
                "--output_path", "/home/resources/automl/output/automl_output_${iteration}.json")
    else
        arrayOf("bash", "-c", "python /automl/main.py" +
                " --dataset " + dataset + " --metric " + metric + " --mode " + mode + " --batch_size " + batchSize + " --seed " + seed +
                " --input_path /home/resources/automl/input/automl_input_${iteration}.json" +
                " --output_path /home/resources/automl/output/automl_output_${iteration}.json")

    val proc = Runtime.getRuntime().exec(exec)
    val stdInput = BufferedReader(InputStreamReader(proc.inputStream))
    val stdError = BufferedReader(InputStreamReader(proc.errorStream))

    println("Here is the standard output of the command:\n")
    var s: String?
    while (stdInput.readLine().also { s = it } != null) {
        println(s)
    }
    println("Here is the standard error of the command (if any):\n")
    while (stdError.readLine().also { s = it } != null) {
        println(s)
    }
}