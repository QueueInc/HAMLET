package org.queueinc.hamlet.controller

import it.unibo.tuprolog.argumentation.core.Arg2pSolver
import it.unibo.tuprolog.argumentation.core.dsl.arg2pScope
import it.unibo.tuprolog.argumentation.core.libs.basic.FlagsBuilder
import it.unibo.tuprolog.core.Struct
import it.unibo.tuprolog.core.parsing.parse
import it.unibo.tuprolog.solve.MutableSolver
import it.unibo.tuprolog.solve.SolveOptions
import it.unibo.tuprolog.solve.TimeDuration
import it.unibo.tuprolog.solve.classic.ClassicSolverFactory
import it.unibo.tuprolog.theory.Theory
import it.unibo.tuprolog.theory.parsing.parse
import org.queueinc.hamlet.argumentation.SpaceGenerator
import org.queueinc.hamlet.argumentation.SpaceMining
import org.queueinc.hamlet.automl.*

val dataset = "vehicle"
val metric = "accuracy"
val mode = "max"
val batchSize = 50
val seed = 42

class Controller(private val dockerMode: Boolean, private val dataManager: FileSystemManager) {

    private val arg2p = Arg2pSolver.default(staticLibs = listOf(SpaceGenerator), dynamicLibs = listOf(SpaceMining))
    private var lastSolver : MutableSolver? = null

    private var config: Config = Config(0)

    private fun nextIteration() = Config(config.iteration + 1)
    private fun updateIteration() {
        config.iteration++
        dataManager.saveConfig(config)
    }

    fun init() {

        if (dockerMode) {
            stopAutoML()
            runAutoML(dataManager.workspacePath)
        }

        config = dataManager.loadConfig() ?: config
    }

    fun stop() {
        if (dockerMode) {
            stopAutoML()
        }
    }

    fun knowledgeBase() : String? = dataManager.loadKnowledgeBase(config.copy())

    fun autoMLData() : AutoMLResults? = dataManager.loadAutoMLData(config.copy())

    fun graphData() : MutableSolver? =
        dataManager.loadGraphData(config.copy())?.let { graph ->
            ClassicSolverFactory.mutableSolverWithDefaultBuiltins(otherLibraries = arg2p.to2pLibraries()).also {
                lastSolver = it
                arg2pScope {
                    it.solve("miner" call "load_graph_data"(Struct.parse(graph))).first()
                }
            }
        }

    fun generateGraph(theory: String, update: (MutableSolver) -> Unit) {

        val creationRules = SpaceGenerator.createGeneratorRules(theory)
        // println(creationRules)
        val solver = ClassicSolverFactory.mutableSolverWithDefaultBuiltins(
            otherLibraries = arg2p.to2pLibraries().plus(FlagsBuilder(
                argumentLabellingMode = "grounded_hash",
                graphExtensions = emptyList()).create().content()),
            staticKb = Theory.parse(theory + "\n" + creationRules, arg2p.operators())
        )

        Thread {
            solver.solve(
                Struct.parse("buildLabelSetsSilent"),
                SolveOptions.allLazilyWithTimeout(TimeDuration.MAX_VALUE)
            ).first()

            update(solver)
            lastSolver = solver
        }.start()
    }

    fun launchAutoML(theory: String, update: (AutoMLResults) -> Unit) {
        lastSolver?.also { solver ->
            Thread {

                dataManager.saveAutoMLData(nextIteration(), solver)
                dataManager.saveKnowledgeBase(nextIteration(), theory)
                dataManager.saveGraphData(nextIteration(), dumpGraphData() ?: "")

                println("Input created for iteration ${nextIteration()}")

                execAutoML(nextIteration().iteration, dockerMode)

                if (dataManager.existsAutoMLData(nextIteration())) {
                    updateIteration()
                    update(dataManager.loadAutoMLData(config.copy())!!)
                } else {
                    // input.delete()
                }
            }.start()

        }
    }

    private fun dumpGraphData() : String? =
        arg2pScope {
            lastSolver?.solve("miner" call "dump_graph_data"(X))
                ?.filter { it.isYes }
                ?.map { it.substitution[X].toString() }
                ?.first()
        }
}