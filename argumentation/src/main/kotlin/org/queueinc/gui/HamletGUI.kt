package org.queueinc.gui

import edu.uci.ics.jung.algorithms.layout.KKLayout
import edu.uci.ics.jung.algorithms.layout.Layout
import edu.uci.ics.jung.graph.Graph
import edu.uci.ics.jung.graph.SparseMultigraph
import edu.uci.ics.jung.graph.util.EdgeType
import edu.uci.ics.jung.visualization.VisualizationViewer
import edu.uci.ics.jung.visualization.control.DefaultModalGraphMouse
import edu.uci.ics.jung.visualization.control.ModalGraphMouse
import edu.uci.ics.jung.visualization.decorators.ToStringLabeller
import edu.uci.ics.jung.visualization.renderers.Renderer
import it.unibo.tuprolog.argumentation.core.Arg2pSolver
import it.unibo.tuprolog.argumentation.core.dsl.arg2pScope
import it.unibo.tuprolog.argumentation.core.libs.basic.FlagsBuilder
import it.unibo.tuprolog.argumentation.core.mining.graph
import it.unibo.tuprolog.argumentation.core.model.Attack
import it.unibo.tuprolog.argumentation.core.model.LabelledArgument
import it.unibo.tuprolog.core.Struct
import it.unibo.tuprolog.core.Term
import it.unibo.tuprolog.core.Var
import it.unibo.tuprolog.core.parsing.parse
import it.unibo.tuprolog.dsl.prolog
import it.unibo.tuprolog.solve.MutableSolver
import it.unibo.tuprolog.solve.SolveOptions
import it.unibo.tuprolog.solve.TimeDuration
import it.unibo.tuprolog.solve.classic.ClassicSolverFactory
import it.unibo.tuprolog.solve.classic.classic
import it.unibo.tuprolog.theory.Theory
import it.unibo.tuprolog.theory.parsing.parse
import it.unibo.tuprolog.unify.Unificator
import javafx.application.Application
import javafx.embed.swing.SwingNode
import javafx.scene.Scene
import javafx.scene.control.Button
import javafx.scene.control.TextArea
import javafx.scene.layout.HBox
import javafx.scene.layout.VBox
import javafx.stage.Stage
import java.awt.Color
import java.awt.Dimension
import java.awt.event.ComponentAdapter
import java.awt.event.ComponentEvent
import java.io.File
import javax.swing.*
import kotlin.system.exitProcess

var path = ""


class FakeMain {

    companion object {

        private val theory: String = FakeMain::class.java.getResource("example.pl").let {
            it!!.readText()
        }

        fun Term.toSklearnClass() =
            when(this.toString()) {
                "dt" -> "DecisionTreeClassifier"
                "knn" -> "KNeighborsClassifier"
                "kbins" -> "KBinsDiscretizer"
                "standard" -> "StandardScaler"
                "minmax" -> "MinMaxScaler"
                "functionTransformer" -> "FunctionTransformer"
                else -> this.toString()
            }

        @JvmStatic
        fun translateSpace(space: Term) =
            prolog {
                space.castToList().toList().joinToString(",\n") { step ->
                    Unificator.default.mgu(step, tupleOf(X, "choice", Z)).let { unifier ->
                        unifier[Z]!!.castToList().toList().map { operator ->
                            if (operator.isAtom) {
                                if (operator.toString() == "functionTransformer") {
                                    """
                                    {
                                        "type" : "FunctionTransformer"
                                    }
                                    """
                                } else "\"$operator\""
                            } else {
                                Unificator.default.mgu(operator, tupleOf(A, B)).let { unifier2 ->
                                    unifier2[B]!!.castToList().toList().joinToString(",\n") { hyper ->
                                        Unificator.default.mgu(hyper, tupleOf(C, D, E)).let { unifier3 ->
                                            """
                                            "${unifier3[C]}" : {
                                               "${unifier3[D]}" : ${unifier3[E]}
                                            }
                                            """
                                        }
                                    }.let {
                                        """
                                        {
                                            "type" : "${unifier2[A]!!.toSklearnClass()}",
                                            $it
                                        }
                                        """
                                    }
                                }
                            }
                        }.let {
                            """
                            "${unifier[X]}" : {
                               "choice" : $it
                            }
                            """
                        }
                    }
                }.let {
                    """
                    "space" : {
                        $it
                    }
                    """.replace("\\s".toRegex(), "")
                }
            }

        @JvmStatic
        fun translateTemplates(mandatory: Term, forbidden: Term) : String {
            val transform = { target: Term, comparator: String ->
                prolog {
                    target.castToList().toList().map { template ->
                        Unificator.default.mgu(template, tupleOf(A, B, C)).let { unifier ->
                            unifier[A]!!.castToList().toList().mapIndexed { i, step ->
                                """
                                "$step" : {"type": {"$comparator": ${unifier[B]!!.castToList().toList()[i].castToList().toList().map { "\"${it.toSklearnClass()}\"" }}}}
                                """
                            }.joinToString(",\n").let {
                                """
                                {
                                    $it,
                                    "classification": {"type": {"eq": "${unifier[C]!!.toSklearnClass()}"}}
                                }
                                """
                            }
                        }
                    }
                }
            }
            return """
                "template_constraints" : ${transform(mandatory, "nin") + transform(forbidden, "in")}
            """.replace("\\s".toRegex(), "")
        }

        @JvmStatic
        fun translateInstances(instances: Term) =
            prolog {
                instances.castToList().toList().map { instance ->
                    instance.castToList().castToList().toList().joinToString(",\n") { step ->
                        if (Unificator.default.match(step, tupleOf("prototype", `_`))) {
                            Unificator.default.mgu(step, tupleOf("prototype", A)).let {
                                "\"prototype\" : \"${it[A]}\""
                            }
                        }
                        else if(Unificator.default.match(step, tupleOf(D, tupleOf(E, F)))) {
                            Unificator.default.mgu(step, tupleOf(D, tupleOf(E, F))).let {
                                it[F]!!.castToList().toList().joinToString(",\n") { h ->
                                    Unificator.default.mgu(h, tupleOf(G, H)).let { hyper ->
                                        "\"${hyper[G]}\" : ${hyper[H]}"
                                    }
                                }.let { hyperparameters ->
                                    """
                                    "${it[D]}": {
                                        "type": "${it[E]!!.toSklearnClass()}",
                                        $hyperparameters
                                    }
                                    """
                                }
                            }
                        }
                        else {
                            Unificator.default.mgu(step, tupleOf(B, C)).let {
                                """
                                "${it[B]}": {
                                    "type": "${it[C]!!.toSklearnClass()}"
                                }
                                """
                            }
                        }
                    }.let {
                        """
                           {
                                $it
                           } 
                        """
                    }
                }.let {
                    """
                        "instance_constraints" : $it 
                    """.replace("\\s".toRegex(), "")
                }
            }

        @JvmStatic
        fun mineData(solver: MutableSolver) =
            arg2pScope {

                val space = solver.solve("miner" call "fetch_complete_space"(X), SolveOptions.allLazilyWithTimeout(TimeDuration.MAX_VALUE))
                    .filter { it.isYes }
                    .map { translateSpace(it.substitution[X]!!) }
                    .first()

                val templates = solver.solve(("miner" call "fetch_mandatory"(X)) and ("miner" call "fetch_forbidden"(Y)), SolveOptions.allLazilyWithTimeout(TimeDuration.MAX_VALUE))
                    .filter { it.isYes }
                    .map { translateTemplates(it.substitution[X]!!, it.substitution[Y]!!) }
                    .first()

                val instances = solver.solve("miner" call "fetch_out_instances"(X), SolveOptions.allLazilyWithTimeout(TimeDuration.MAX_VALUE))
                    .filter { it.isYes }
                    .map { translateInstances(it.substitution[X]!!) }
                    .first()

                "{$space,$templates,$instances,\"points_to_evaluate\":[],\"evaluated_rewards\":[]}"
            }

        @JvmStatic
        fun main(args: Array<String>) {
            val arg2p = Arg2pSolver.default(staticLibs = listOf(HamletCore), dynamicLibs = listOf(SpaceMining))
            val solver = ClassicSolverFactory.mutableSolverWithDefaultBuiltins(
                otherLibraries = arg2p.to2pLibraries().plus(FlagsBuilder(graphExtensions = emptyList()).create().content()),
                staticKb = Theory.parse(theory, arg2p.operators())
            )

            arg2pScope {
                solver.solve("buildLabelSets"(X, Y), SolveOptions.allLazilyWithTimeout(TimeDuration.MAX_VALUE))
                    .filter { it.isYes }
                    .first()
                //    .forEach { println(it.substitution[X]) }

                println(mineData(solver))
            }
        }
    }
}

class HamletGUI : Application() {

    override fun start(stage: Stage) {
        try {

            val arg2p = Arg2pSolver.default(staticLibs = emptyList(), dynamicLibs = emptyList())
            val graph = ArgumentationGraphFrame.customTab()
            var graphResult : it.unibo.tuprolog.argumentation.core.model.Graph? = null

            stage.title = "HAMLET"

            val textArea = TextArea(File("$path/kb.txt").readText())

            val compute = Button("Compute Graph")
            compute.minWidth = 50.0

            compute.setOnAction { _ ->
                val solver = ClassicSolverFactory.mutableSolverWithDefaultBuiltins(
                    otherLibraries = arg2p.to2pLibraries().plus(FlagsBuilder().create().content()),
                    staticKb = Theory.parse(textArea.text, arg2p.operators())
                )
                solver.solve(Struct.parse("buildLabelSets(_, _)"), SolveOptions.allLazilyWithTimeout(TimeDuration.MAX_VALUE))
                    .first()
                    .also { graphResult = graph.update(solver) }
            }

            val export = Button("Export")
            export.minWidth = 50.0

            export.setOnAction { _ ->
                val res = graphResult?.labellings?.map {
                    "${it.label} - ${it.argument.descriptor}"
                }?.joinToString(separator = "\n")
                File("$path/graph.txt").writeText(res ?: "")
                File("$path/kb.txt").writeText(textArea.text)
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
        exitProcess(0)
    }

    companion object {
        @JvmStatic
        fun main(args: Array<String>) {
            path = args[0]
            launch(HamletGUI::class.java)
        }
    }
}

data class GraphNode(val node : SwingNode, val update : (MutableSolver) -> it.unibo.tuprolog.argumentation.core.model.Graph)

internal class ArgumentationGraphFrame {

    private val graphPane: JScrollPane = JScrollPane()
    private val classicTheoryPane: JScrollPane = JScrollPane()
    private val treeTheoryPane: JScrollPane = JScrollPane()
    val splitPane: JSplitPane = JSplitPane(JSplitPane.HORIZONTAL_SPLIT)

    private val next: JButton = JButton("Next").also { button ->
        button.addActionListener {
            this.selectedContext =
                if (this.selectedContext + 1 >= this.maxContext) this.maxContext else this.selectedContext + 1
            this.update()
        }
    }
    private val back: JButton = JButton("Back").also { button ->
        button.addActionListener {
            this.selectedContext =
                if (this.selectedContext - 1 <= this.minContext) this.minContext else this.selectedContext - 1
            this.update()
        }
    }
    private val context: JLabel = JLabel()

    private val minContext: Int = 0
    private var maxContext: Int = 0
    private var selectedContext: Int = 0

    private var mutableSolver: MutableSolver? = null

    init {

        val panel = JPanel()
        panel.layout = BoxLayout(panel, BoxLayout.Y_AXIS)

        val buttonPanel = JPanel()
        buttonPanel.layout = BoxLayout(buttonPanel, BoxLayout.X_AXIS)
        buttonPanel.add(back)
        buttonPanel.add(next)
        buttonPanel.add(context)

        val tabbedPane = JTabbedPane()
        tabbedPane.addTab("Classic", classicTheoryPane)
        tabbedPane.addTab("Tree", treeTheoryPane)

        panel.add(tabbedPane)
        panel.add(buttonPanel)

        splitPane.add(panel)
        splitPane.add(graphPane)

        splitPane.isOneTouchExpandable = true
        splitPane.dividerLocation = 150
    }

    private fun update() {
        SwingUtilities.invokeLater {
            back.isEnabled = this.selectedContext > this.minContext
            next.isEnabled = this.selectedContext < this.maxContext
            context.text = this.selectedContext.toString()
        }
        mutableSolver?.also { solver ->
            try {
                val graph = solver.graph(this.selectedContext)
                SwingUtilities.invokeLater {
                    printGraph(this.graphPane, graph.labellings, graph.attacks)
                    printTheory(this.classicTheoryPane, this.treeTheoryPane, graph.labellings)
                }
            } catch (e: Exception) {
                this.clear()
            }
        } ?: clear()
        revalidate()
    }

    private fun clear() {
        SwingUtilities.invokeLater {
            this.graphPane.viewport.removeAll()
            this.classicTheoryPane.viewport.removeAll()
            this.treeTheoryPane.viewport.removeAll()
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
            val frame = ArgumentationGraphFrame()
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
                frame.maxContext = frame.selectedContext
                frame.update()
                model.graph(frame.selectedContext)
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
        private fun printGraph(graphPane: JScrollPane, arguments: List<LabelledArgument>, attacks: List<Attack>) {
            val layout: Layout<String, String> = KKLayout(buildGraph(arguments, attacks))
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
            graphPane.viewport.view = vv
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

