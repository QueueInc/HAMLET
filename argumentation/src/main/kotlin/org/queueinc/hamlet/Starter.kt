package org.queueinc.hamlet

import it.unibo.tuprolog.solve.MutableSolver
import javafx.application.Application
import javafx.stage.Stage
import org.queueinc.hamlet.controller.AutoMLResults
import org.queueinc.hamlet.controller.Controller
import org.queueinc.hamlet.controller.FileSystemManager
import org.queueinc.hamlet.gui.GUI
import kotlin.system.exitProcess

private var path : String = ""
private var debugMode : Boolean = true

object Starter {
    @JvmStatic
    fun main(args: Array<String>) {
        path = args[0]
        debugMode = if (args.size == 1) false else args[1].toBoolean()
        Application.launch(HAMLET::class.java)
    }
}

class HAMLET : Application() {

    private val controller = Controller(debugMode, FileSystemManager(path))

    override fun start(stage: Stage) {
        try {

            controller.init()
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