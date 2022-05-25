package org.queueinc.hamlet

import it.unibo.tuprolog.solve.MutableSolver
import javafx.application.Application
import javafx.stage.Stage
import org.queueinc.hamlet.controller.Controller
import org.queueinc.hamlet.gui.AutoMLResults
import org.queueinc.hamlet.gui.GUI
import kotlin.system.exitProcess

private var path : String = ""
private var dockerMode : Boolean = true

object Starter {
    @JvmStatic
    fun main(args: Array<String>) {
        path = args[0]
        dockerMode = args[1].toBoolean()
        Application.launch(HAMLET::class.java)
    }
}

class HAMLET : Application() {

    private val controller = Controller(path, dockerMode)

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
            controller.loadKnowledgeBase()?.also { view.displayTheory(it) }
            controller.loadAutoMLData()?.also { view.displayAutoMLData(it) }
            controller.loadGraphData()?.also { view.displayGraph(it) }
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