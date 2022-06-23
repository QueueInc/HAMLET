package org.queueinc.hamlet

import it.unibo.tuprolog.solve.MutableSolver
import javafx.application.Application
import javafx.stage.Stage
import org.queueinc.hamlet.controller.AutoMLResults
import org.queueinc.hamlet.controller.Controller
import org.queueinc.hamlet.controller.FileSystemManager
import org.queueinc.hamlet.gui.GUI
import java.io.File
import kotlin.system.exitProcess

private var path: String = ""
private var dataset: String = ""
private var metric: String = ""
private var mode: String = ""
private var batchSize: Int = 0
private var seed: Int = 0
private var debugMode: Boolean = true
private var theory: String = ""

object Starter {
    @JvmStatic
    fun main(args: Array<String>) {
        path = args[0]
        dataset = args[1]
        metric = args[2]
        mode = args[3]
        batchSize = args[4].toInt()
        seed = args[5].toInt()
        debugMode = args[6].toBoolean()
        theory = if (args.size == 8) File(args[7]).readText() else ""

        if (theory == "") {
            Application.launch(HAMLET::class.java)
        }
        else {
            consoleHamlet()
        }
    }
}


fun consoleHamlet() {
    Controller(debugMode, FileSystemManager(path)).also { controller ->
        controller.init(dataset, metric, mode, batchSize, seed)
        controller.generateGraph(theory) {}
        controller.launchAutoML(theory) {}
        controller.stop()
    }
}

class HAMLET : Application() {

    private val controller = Controller(debugMode, FileSystemManager(path))

    override fun start(stage: Stage) {
        try {

            controller.init(dataset, metric, mode, batchSize, seed)
            val computeAction : (String, (MutableSolver) -> Unit) -> Unit = { kb, updateAction ->
                controller.generateGraph(kb, updateAction)
            }
            val exportAction : (String, (AutoMLResults) -> Unit) -> Unit = { kb, exportAction ->
                controller.launchAutoML(kb, exportAction)
            }

            val view = GUI(stage)
            view.prepareStage(computeAction, exportAction)
            controller.knowledgeBase()?.also { view.displayTheory(it) }
            controller.autoMLData()?.also { view.displayAutoMLData(it) }
            controller.graphData()?.also { view.displayGraph(it) }
            view.show()

        } catch (e: Throwable) {
            e.printStackTrace()
            throw Error(e)
        }
    }

    override fun stop() {
        controller.stop()
        exitProcess(0)
    }
}