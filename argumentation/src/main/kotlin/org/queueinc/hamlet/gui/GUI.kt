package org.queueinc.hamlet.gui

import it.unibo.tuprolog.solve.MutableSolver
import javafx.application.Platform
import javafx.geometry.Orientation
import javafx.scene.Scene
import javafx.scene.control.*
import javafx.scene.layout.HBox
import javafx.scene.layout.Priority
import javafx.scene.layout.VBox
import javafx.stage.Stage


class GUI(private val stage: Stage) {

    fun prepareStage(theory: String, computeAction : (String, (MutableSolver) -> Unit) -> Unit, exportAction : (String) -> Unit) {

        val graph = GraphVisualizer.customTab()
        val textArea = TextArea(theory)
        val compute = Button("Compute Graph")
        val export = Button("Run AutoML")

        val tabPane = TabPane()
        val tab1 = Tab("Graph", graph.node)
        tabPane.tabs.add(tab1)

        val vbox = VBox(HBox(compute, export), tabPane)
        VBox.setVgrow(tabPane, Priority.ALWAYS)

        graph.node.maxHeight(Double.MAX_VALUE)


        val splitPane = SplitPane()
        splitPane.orientation = Orientation.VERTICAL
        splitPane.items.addAll(textArea, vbox)

        val scene = Scene(splitPane, 700.0, 600.0)

        compute.minWidth = 50.0
        export.minWidth = 50.0

        compute.setOnAction {
            computeAction(textArea.text) {
                Platform.runLater { graph.update(it) }
            }
        }

        export.setOnAction {
            exportAction(textArea.text)
        }

        stage.title = "HAMLET"
        stage.scene = scene
    }

    fun show() = stage.show()
}
