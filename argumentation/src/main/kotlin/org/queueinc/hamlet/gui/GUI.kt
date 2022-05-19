package org.queueinc.hamlet.gui

import it.unibo.tuprolog.solve.MutableSolver
import javafx.application.Platform
import javafx.beans.property.ReadOnlyObjectWrapper
import javafx.collections.FXCollections
import javafx.collections.ObservableList
import javafx.geometry.Orientation
import javafx.scene.Scene
import javafx.scene.control.*
import javafx.scene.layout.HBox
import javafx.scene.layout.Priority
import javafx.scene.layout.VBox
import javafx.stage.Stage

data class AutoMLResults(val values: List<List<String>>)

class GUI(private val stage: Stage) {

    fun prepareStage(theory: String, computeAction : (String, (MutableSolver) -> Unit) -> Unit, exportAction : (String, (AutoMLResults) -> Unit) -> Unit) {

        val graph = GraphVisualizer.customTab()
        val textArea = TextArea(theory)
        val compute = Button("Compute Graph")
        val export = Button("Run AutoML")

        val tabPane = TabPane()
        val tab1 = Tab("Graph", graph.node)
        val tab2 = Tab("Data")
        tabPane.tabs.add(tab1)
        tabPane.tabs.add(tab2)

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
            exportAction(textArea.text) {
                Platform.runLater { tab2.content = toTableView(it.values) }
            }
        }

        stage.title = "HAMLET"
        stage.scene = scene
    }

    fun show() = stage.show()
}

fun toTableView(rows: List<List<String>>) : TableView<ObservableList<String>> {
    val tableView = TableView<ObservableList<String>>()
    for (i in rows[0].indices) {
        val column: TableColumn<ObservableList<String>, String> = TableColumn(
            rows[0][i]
        )
        column.setCellValueFactory { param -> ReadOnlyObjectWrapper(param.value[i]) }
        tableView.columns.add(column)
    }

    rows.drop(1).forEach {
        tableView.items.add(
            FXCollections.observableArrayList(it)
        )
    }

    return tableView
}

