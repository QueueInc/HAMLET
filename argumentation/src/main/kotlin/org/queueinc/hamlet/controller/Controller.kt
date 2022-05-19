package org.queueinc.hamlet.controller

import com.google.gson.Gson
import it.unibo.tuprolog.argumentation.core.Arg2pSolver
import it.unibo.tuprolog.argumentation.core.libs.basic.FlagsBuilder
import it.unibo.tuprolog.core.Struct
import it.unibo.tuprolog.core.parsing.parse
import it.unibo.tuprolog.solve.MutableSolver
import it.unibo.tuprolog.solve.SolveOptions
import it.unibo.tuprolog.solve.TimeDuration
import it.unibo.tuprolog.solve.classic.ClassicSolverFactory
import it.unibo.tuprolog.theory.Theory
import it.unibo.tuprolog.theory.parsing.parse
import org.queueinc.hamlet.argumentation.SpaceGenerator
import org.queueinc.hamlet.argumentation.SpaceMining
import org.queueinc.hamlet.createAndWrite
import java.io.BufferedReader
import java.io.File
import java.io.InputStreamReader

val dataset = "wine"
val metric = "accuracy"
val mode = "max"
val batchSize = 25
val seed = 42

class Controller(private val workspacePath: String, private val pythonMode: Boolean) {

    private val arg2p = Arg2pSolver.default(staticLibs = listOf(SpaceGenerator), dynamicLibs = listOf(SpaceMining))
    private var lastSolver : MutableSolver? = null
    private lateinit var configReference: File

    private var config: Config = Config(0)
    private var oldKb = ""

    fun init() : String {

        if (!pythonMode) {
            stopAutoML()
            runAutoML()
        }

        configReference = File("$workspacePath/config.json")
        if (configReference.exists()) {
            config = Gson().fromJson(configReference.readText(), Config::class.java)
            oldKb = File("$workspacePath/argumentation/kb_${config.iteration}.txt").readText()
        }
        else {
            File("$workspacePath/argumentation").mkdirs()
            File("${workspacePath}/automl/input").mkdirs()
            File("${workspacePath}/automl/output").mkdirs()
        }

        return oldKb
    }

    fun stop() {
        if (!pythonMode) {
            stopAutoML()
        }
    }

    fun generateGraph(theory: String, update: (MutableSolver) -> Unit) {
        val solver = ClassicSolverFactory.mutableSolverWithDefaultBuiltins(
            otherLibraries = arg2p.to2pLibraries().plus(FlagsBuilder(graphExtensions = emptyList()).create().content()),
            staticKb = Theory.parse(theory, arg2p.operators())
        )
        Thread {
            solver.solve(
                Struct.parse("buildLabelSets(_, _)"),
                SolveOptions.allLazilyWithTimeout(TimeDuration.MAX_VALUE)).first()
            update(solver)
            lastSolver = solver
        }.start()
    }

    fun launchAutoML(theory: String) {
        lastSolver?.also {
            val res = SpaceTranslator.mineData(it)
            config.iteration++
            File("${workspacePath}/automl/input/automl_input_${config.iteration}.json").createAndWrite(res)
            File("${workspacePath}/argumentation/kb_${config.iteration}.txt").createAndWrite(theory)
            configReference.createAndWrite(Gson().toJson(config))

            Thread {
                execAutoML(config.iteration, pythonMode)
            }.start()

        }
    }

    private fun stopAutoML() {
        val stop = arrayOf("bash", "-c", "docker stop automl_container")
        val rm = arrayOf("bash", "-c", "docker rm automl_container")

        Runtime.getRuntime().exec(stop).waitFor()
        Runtime.getRuntime().exec(rm).waitFor()
    }

    private fun runAutoML() {

        // Little pig
        val tempPath = if (workspacePath.startsWith("c:", ignoreCase = true))
            workspacePath.replace("c:", "/mnt/c", ignoreCase = true)
        else workspacePath

        val build = arrayOf("bash", "-c", "docker build -t automl_container ./.devcontainer")
        val run = arrayOf("bash", "-c", "docker run --name automl_container --volume \$(pwd)/automl:/home/automl --volume ${tempPath}:/home/resources --detach -t automl_container")

        Runtime.getRuntime().exec(build).waitFor()
        Runtime.getRuntime().exec(run).waitFor()
    }

    private fun execAutoML(iteration: Int, pythonMode: Boolean) {

        val exec  = if (!pythonMode)
            arrayOf("bash", "-c", "docker exec automl_container python automl/main.py" +
                " --dataset " + dataset + " --metric " + metric + " --mode " + mode + " --batch_size " + batchSize + " --seed " + seed +
                " --input_path /home/resources/automl/input/automl_input_${iteration}.json" +
                " --output_path /home/resources/automl/output/automl_output_${iteration}.json")
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

}