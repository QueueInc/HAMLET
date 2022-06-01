package org.queueinc.hamlet.gui

import edu.uci.ics.jung.algorithms.layout.FRLayout
import edu.uci.ics.jung.algorithms.layout.Layout
import edu.uci.ics.jung.graph.Graph
import edu.uci.ics.jung.graph.SparseMultigraph
import edu.uci.ics.jung.graph.util.EdgeType
import edu.uci.ics.jung.visualization.GraphZoomScrollPane
import edu.uci.ics.jung.visualization.VisualizationViewer
import edu.uci.ics.jung.visualization.control.DefaultModalGraphMouse
import edu.uci.ics.jung.visualization.control.ModalGraphMouse
import edu.uci.ics.jung.visualization.decorators.ToStringLabeller
import edu.uci.ics.jung.visualization.renderers.Renderer
import it.unibo.tuprolog.argumentation.core.mining.graph
import it.unibo.tuprolog.argumentation.core.model.Attack
import it.unibo.tuprolog.argumentation.core.model.LabelledArgument
import it.unibo.tuprolog.dsl.prolog
import it.unibo.tuprolog.solve.MutableSolver
import it.unibo.tuprolog.solve.classic.classic
import javafx.embed.swing.SwingNode
import java.awt.BorderLayout
import java.awt.Color
import java.awt.Dimension
import java.awt.event.ComponentAdapter
import java.awt.event.ComponentEvent
import javax.swing.*

internal class GraphVisualizer {

    private val graphPane: JPanel = JPanel(BorderLayout())
    private val classicTheoryPane: JScrollPane = JScrollPane()
    private val treeTheoryPane: JScrollPane = JScrollPane()
    val splitPane: JSplitPane = JSplitPane(JSplitPane.HORIZONTAL_SPLIT)

    private var selectedContext: Int = 0

    private var mutableSolver: MutableSolver? = null

    init {

        val panel = JPanel()
        panel.layout = BoxLayout(panel, BoxLayout.Y_AXIS)

        val tabbedPane = JTabbedPane()
        tabbedPane.addTab("Classic", classicTheoryPane)
        tabbedPane.addTab("Tree", treeTheoryPane)

        panel.add(tabbedPane)

        splitPane.add(panel)
        splitPane.add(graphPane)

        splitPane.isOneTouchExpandable = true
        splitPane.dividerLocation = 150
    }

    private fun update() {
        mutableSolver?.also { solver ->
            Thread {
                try {
                    val graph = solver.graph(this.selectedContext)
                    SwingUtilities.invokeLater {
                        this.graphPane.removeAll()
                        this.classicTheoryPane.viewport.removeAll()
                        this.treeTheoryPane.viewport.removeAll()
                        printGraph(this.graphPane, graph.labellings, graph.attacks)
                        printTheory(this.classicTheoryPane, this.treeTheoryPane, graph.labellings)
                        this.splitPane.revalidate()
                    }
                } catch (e: Exception) {
                    this.clear()
                }
            }.start()
        } ?: clear()
    }

    private fun clear() {
        SwingUtilities.invokeLater {
            this.graphPane.removeAll()
            this.classicTheoryPane.viewport.removeAll()
            this.treeTheoryPane.viewport.removeAll()
            revalidate()
        }
    }

    private fun revalidate() {
        SwingUtilities.invokeLater {
            this.splitPane.repaint()
        }
    }
    companion object {

        @JvmStatic
        fun customTab() : GraphNode {
            val frame = GraphVisualizer()
            val swingNode = SwingNode()
            frame.splitPane.addComponentListener(object : ComponentAdapter() {
                override fun componentResized(e: ComponentEvent) {
                    frame.revalidate()
                }
            })
            swingNode.content = frame.splitPane
            frame.update()
            return GraphNode(swingNode) { model ->
                frame.mutableSolver = MutableSolver.classic(
                    libraries = model.libraries
                )
                frame.selectedContext = prolog {
                    frame.mutableSolver!!.solve("context_active"(X))
                        .map { it.substitution[X]!!.asNumeric()!!.intValue.toInt() }
                        .first()
                }
                frame.update()
            }
        }

        @JvmStatic
        private fun buildGraph(arguments: List<LabelledArgument>, attacks: List<Attack>): Graph<String, String> {
            val graph: Graph<String, String> = SparseMultigraph()
            arguments.map { it.argument.identifier }
                .forEach(graph::addVertex)
            attacks.forEach { x ->
                graph.addEdge(
                    x.attacker.identifier + x.target.identifier,
                    x.attacker.identifier,
                    x.target.identifier,
                    EdgeType.DIRECTED
                )
            }
            return graph
        }

        @JvmStatic
        private fun printGraph(graphPane: JPanel, arguments: List<LabelledArgument>, attacks: List<Attack>) {
            val layout: Layout<String, String> = FRLayout(buildGraph(arguments, attacks))
            layout.size = Dimension(350, 300)
            val vv: VisualizationViewer<String, String> = VisualizationViewer(layout)
            vv.preferredSize = Dimension(350, 300)
            vv.renderContext.setVertexFillPaintTransformer { i ->
                when (arguments.first { x -> x.argument.identifier == i }.label) {
                    "in" -> Color.GREEN
                    "out" -> Color.RED
                    else -> Color.GRAY
                }
            }
            vv.renderContext.vertexLabelTransformer = ToStringLabeller()
            vv.renderer.vertexLabelRenderer.position = Renderer.VertexLabel.Position.AUTO
            val graphMouse: DefaultModalGraphMouse<String, String> = DefaultModalGraphMouse()
            graphMouse.setMode(ModalGraphMouse.Mode.PICKING)
            vv.graphMouse = graphMouse
            vv.addKeyListener(graphMouse.modeKeyListener)
            graphPane.add(GraphZoomScrollPane(vv), BorderLayout.CENTER)
        }

        @JvmStatic
        private fun printTheory(classicTheoryPane: JScrollPane, treeTheoryPane: JScrollPane, arguments: List<LabelledArgument>) {
            val textArea = JTextArea()
            textArea.isEditable = false
            arguments.sortedBy { it.argument.identifier.drop(1).toInt() }
                .forEach { x -> textArea.append(x.argument.descriptor + "\n") }
            classicTheoryPane.viewport.view = textArea

            val textAreaTree = JTextPane()
            textAreaTree.isEditable = false
            textAreaTree.contentType = "text/html"
            textAreaTree.text = formatResolutionTree(arguments)
            treeTheoryPane.viewport.view = textAreaTree
        }

        @JvmStatic
        private fun formatResolutionTree(arguments: List<LabelledArgument>): String {
            fun tree(arg: LabelledArgument, arguments: List<LabelledArgument>): String =
                "<li>${arg.argument.descriptor} <b>[${arg.label.uppercase()}]</b></li>" +
                        arg.argument.supports.joinToString(separator = "") { sub ->
                            tree(
                                arguments.first { it.argument.identifier == sub.identifier },
                                arguments
                            )
                        }.let { if (it.isNotEmpty()) "<ul>$it</ul>" else it }

            return "<html><ul>" + arguments
                .sortedBy { it.argument.identifier.drop(1).toInt() }
                .joinToString(separator = "") { tree(it, arguments) } + "</ul></html>"
        }
    }
}
