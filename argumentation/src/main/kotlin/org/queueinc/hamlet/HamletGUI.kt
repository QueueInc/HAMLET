package org.queueinc.hamlet

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
import javafx.application.Application
import javafx.application.Platform
import javafx.scene.Scene
import javafx.scene.control.Button
import javafx.scene.control.TextArea
import javafx.scene.layout.HBox
import javafx.scene.layout.VBox
import javafx.stage.Stage
import java.io.BufferedReader
import java.io.File
import java.io.InputStreamReader
import kotlin.system.exitProcess


var workspacePath = ""

data class Config(var iteration : Int)

fun File.createAndWrite(text: String) {
    this.createNewFile()
    this.writeText(text)
}

val dataset = "wine"
val metric = "accuracy"
val mode = "max"
val batchSize = 25
val seed = 42


fun stopAutoML() {
    val stop = arrayOf("bash", "-c", "docker stop automl_container")
    val rm = arrayOf("bash", "-c", "docker rm automl_container")

    Runtime.getRuntime().exec(stop).waitFor()
    Runtime.getRuntime().exec(rm).waitFor()
}

fun runAutoML() {
    val build = arrayOf("bash", "-c", "docker build -t automl_container ./.devcontainer")
    val run = arrayOf("bash", "-c", "docker run --name automl_container --volume \$(pwd)/automl:/home/automl --volume /mnt/c/Users/giuseppe.pisano5/Documents/MyProjects/HAMLET/resources:/home/resources --detach -t automl_container")

    Runtime.getRuntime().exec(build).waitFor()
    Runtime.getRuntime().exec(run).waitFor()
}

fun execAutoML(iteration: Int) {
    val exec = arrayOf("bash", "-c", "docker exec automl_container python automl/main.py" +
        " --dataset " + dataset + " --metric " + metric + " --mode " + mode + " --batch_size " + batchSize + " --seed " + seed +
        " --input_path /home/resources/automl/input/automl_input_${iteration}.json" +
        " --output_path /home/resources/automl/input/automl_output_${iteration}.json")

    val proc = Runtime.getRuntime().exec(exec)
    val stdInput = BufferedReader(InputStreamReader(proc.inputStream))
    val stdError = BufferedReader(InputStreamReader(proc.errorStream))

    println("Here is the standard output of the command:\n")
    var s: String? = null
    while (stdInput.readLine().also { s = it } != null) {
        println(s)
    }
    println("Here is the standard error of the command (if any):\n")
    while (stdError.readLine().also { s = it } != null) {
        println(s)
    }
}



class HamletGUI : Application() {

    private var config: Config = Config(0)
    var oldKb = ""

    override fun start(stage: Stage) {
        try {
            stopAutoML()
            runAutoML()
            println("ciao")
            val configReference = File("$workspacePath/config.json")
            if (configReference.exists()) {
                config = Gson().fromJson(configReference.readText(), Config::class.java)
                oldKb = File("$workspacePath/argumentation/kb_${config.iteration}.txt").readText()
            }
            else {
                File("$workspacePath/argumentation").mkdirs()
                File("${workspacePath}/automl/input").mkdirs()
                File("${workspacePath}/automl/output").mkdirs()
            }

            val arg2p = Arg2pSolver.default(staticLibs = listOf(HamletCore), dynamicLibs = listOf(SpaceMining))
            val graph = GraphVisualizer.customTab()
            var lastSolver : MutableSolver? = null

            stage.title = "HAMLET"

            val textArea = TextArea(oldKb)

            val compute = Button("Compute Graph")
            compute.minWidth = 50.0

            compute.setOnAction { _ ->
                val solver = ClassicSolverFactory.mutableSolverWithDefaultBuiltins(
                    otherLibraries = arg2p.to2pLibraries().plus(FlagsBuilder(graphExtensions = emptyList()).create().content()),
                    staticKb = Theory.parse(textArea.text, arg2p.operators())
                )
                Thread {
                    solver.solve(Struct.parse("buildLabelSets(_, _)"),
                        SolveOptions.allLazilyWithTimeout(TimeDuration.MAX_VALUE)).first()
                    Platform.runLater { graph.update(solver) }
                    lastSolver = solver
                }.start()
            }

            val export = Button("Run")
            export.minWidth = 50.0

            export.setOnAction { _ ->
                lastSolver?.also {
                    val res = SpaceTranslator.mineData(it)
                    config.iteration++
                    File("${workspacePath}/automl/input/automl_input_${config.iteration}.json").createAndWrite(res)
                    File("${workspacePath}/argumentation/kb_${config.iteration}.txt").createAndWrite(textArea.text)
                    configReference.createAndWrite(Gson().toJson(config))

                    Thread {
                        execAutoML(config.iteration)
                    }.start()

                }
            }

            val vbox = VBox(textArea, HBox(compute, export), graph.node)

            val scene = Scene(vbox, 1000.0, 800.0)
            stage.scene = scene
            stage.show()

        } catch (e: Throwable) {
            e.printStackTrace()
            throw Error(e)
        }
    }

    override fun stop() {
        stopAutoML()
        exitProcess(0)
    }
}
