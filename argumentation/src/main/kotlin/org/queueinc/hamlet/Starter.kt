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
private var fairnessMetric: String = ""
private var mode: String = ""
private var batchSize: Int = 0
private var timeBudget: Int = 0
private var seed: Int = 0
private var debugMode: Boolean = true
private var theory: String = ""
private var volume: String? = null

object Starter {
    @JvmStatic
    fun main(args: Array<String>) {
        path = args[0]
        dataset = args[1]
        metric = args[2]
        fairnessMetric = args[3]
        mode = args[4]
        batchSize = args[5].toInt()
        timeBudget = args[6].toInt()
        seed = args[7].toInt()
        debugMode = args[8].toBoolean()
        volume = if (args[9] != "None") args[9] else null
        theory = if (args.size == 11) File(args[10]).readText() else ""

        if (theory == "") {
            Application.launch(HAMLET::class.java)
        }
        else {
            consoleHamlet()
        }
    }
}


fun consoleHamlet() {
    Controller(debugMode, FileSystemManager(path, volume)).also { controller ->
        controller.init(dataset, metric, fairnessMetric, mode, batchSize, timeBudget, seed)
        controller.generateGraph(theory, true) {}
        controller.launchAutoML(true) {}
        controller.stop()
    }
}

class HAMLET : Application() {

    private val controller = Controller(debugMode, FileSystemManager(path, volume))

    override fun start(stage: Stage) {
        try {

            controller.init(dataset, metric, fairnessMetric, mode, batchSize, timeBudget, seed)
            val computeAction : (String, (MutableSolver) -> Unit) -> Unit = { kb, updateAction ->
                controller.generateGraph(kb, false, updateAction)
            }
            val exportAction : (String, (AutoMLResults) -> Unit) -> Unit = { _, exportAction ->
                controller.launchAutoML(false, exportAction)
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