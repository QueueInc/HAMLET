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
import org.queueinc.hamlet.controller.AutoMLResults
import org.queueinc.hamlet.controller.Rule

class GUI(private val stage: Stage) {

    private val graph = GraphVisualizer.customTab()
    private val textArea = TextArea()
    private val compute = Button("Compute Graph")
    private val export = Button("Run AutoML")

    private val tabPane = TabPane()
    private val tab1 = Tab("Graph", graph.node)
    private val tab2 = Tab("Data")
    private val tab3 = Tab("AutoML Arguments")


    fun displayTheory(theory: String) {
        Platform.runLater { textArea.text = theory }
    }

    fun displayAutoMLData(data: AutoMLResults) {
        Platform.runLater {
            tab2.content = toTableView(data.evaluatedPoints)
            tab3.content = toTableView(rulesToTable(data.inferredRules))
        }
    }

    fun displayGraph(solver: MutableSolver) {
        Platform.runLater { graph.update(solver) }
    }

    fun prepareStage(computeAction : (String, (MutableSolver) -> Unit) -> Unit, exportAction : (String, (AutoMLResults) -> Unit) -> Unit) {

        tabPane.tabs.add(tab1)
        tabPane.tabs.add(tab2)
        tabPane.tabs.add(tab3)

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
                graph.update(it)
            }
        }

        export.setOnAction {
            exportAction(textArea.text) {
                Platform.runLater {
                    tab2.content = toTableView(it.evaluatedPoints)
                    tab3.content = toTableView(rulesToTable(it.inferredRules))
                }
            }
        }

        stage.title = "HAMLET"
        stage.scene = scene
    }

    fun show() = stage.show()
}

private fun toTableView(rows: List<List<String>>) : TableView<ObservableList<String>> {
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


private fun rulesToTable(rules: List<Rule>) : List<List<String>> =
    listOf(listOf("argument", "support", "accuracy")) +
            rules
                .filter { it.isValid }
                .map {
                    listOf(it.theoryRepresentation, it.support.toString(), it.metric_threshold.toString())
                }
