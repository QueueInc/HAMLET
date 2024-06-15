package org.queueinc.hamlet.controller

import it.unibo.tuprolog.argumentation.core.Arg2pSolver
import it.unibo.tuprolog.argumentation.core.dsl.arg2pScope
import it.unibo.tuprolog.argumentation.core.libs.basic.FlagsBuilder
import it.unibo.tuprolog.core.Clause
import it.unibo.tuprolog.core.Struct
import it.unibo.tuprolog.core.parsing.parse
import it.unibo.tuprolog.dsl.prolog
import it.unibo.tuprolog.solve.MutableSolver
import it.unibo.tuprolog.solve.SolveOptions
import it.unibo.tuprolog.solve.TimeDuration
import it.unibo.tuprolog.solve.classic.ClassicSolverFactory
import it.unibo.tuprolog.theory.Theory
import it.unibo.tuprolog.theory.parsing.parse
import org.queueinc.hamlet.argumentation.SpaceGenerator
import org.queueinc.hamlet.argumentation.SpaceMining
import org.queueinc.hamlet.automl.*
import kotlin.random.Random


class Controller(private val debugMode: Boolean, private val dataManager: FileSystemManager) {

    private val arg2p = Arg2pSolver.default(staticLibs = listOf(SpaceGenerator), dynamicLibs = listOf(SpaceMining))
    private var lastSolver : MutableSolver? = null
    private var theory = ""

    private var generationTime = 0L

    private lateinit var config: Config

    private fun nextIteration() = config.copy(iteration = config.iteration + 1)
    private fun updateIteration() {
        config.iteration++
        dataManager.saveConfig(config)
    }

    fun init(dataset: String, metric: String, fairnessMetric: String, mode: String, batchSize: Int, timeBudget: Int, seed: Int) {
        // stopAutoML()
        runAutoML(dataManager.workspacePath, dataManager.volume, debugMode)

        config = dataManager.loadConfig().let {
            if (it == null || it.dataset != dataset) {
                dataManager.cleanWorkspace()
                dataManager.initWorkspace()
                Config(0, dataset, metric, fairnessMetric, mode, batchSize, timeBudget, seed)
            }
            else it.copy(metric = metric, fairnessMetric = fairnessMetric, mode = mode, batchSize = batchSize, timeBudget = timeBudget, seed = seed)
        }
    }

    fun stop() {
        stopAutoML()
    }

    fun knowledgeBase() : String? = dataManager.loadKnowledgeBase(config.copy())

    fun autoMLData() : AutoMLResults? = dataManager.loadAutoMLData(config.copy())

    fun graphData() : MutableSolver? =
        dataManager.loadGraphData(config.copy())?.let { graph ->
            theory = dataManager.loadKnowledgeBase(config.copy()) ?: ""
            ClassicSolverFactory.mutableSolverWithDefaultBuiltins(
                otherLibraries = arg2p.to2pLibraries(),
                staticKb = Theory.parse(theory, arg2p.operators())
            ).also {
                lastSolver = it
                arg2pScope {
                    it.solve("miner" call "load_graph_data"(Struct.parse(graph))).first()
                }
            }
        }

    fun generateGraph(theory: String, blocking: Boolean, update: (MutableSolver) -> Unit) {

        this.theory = theory
        val start = System.currentTimeMillis() / 1000
        val creationRules = SpaceGenerator.createGeneratorRules(theory)
        println(creationRules)
        val solver = ClassicSolverFactory.mutableSolverWithDefaultBuiltins(
            otherLibraries = arg2p.to2pLibraries().plus(FlagsBuilder(
                argumentLabellingMode = "grounded_hash",
                graphExtensions = emptyList()).create().content()),
            staticKb = Theory.parse(theory + "\n" + creationRules, arg2p.operators()),
        ).let { solver ->
            prolog {
                solver.solve("preparePipelines"(X)).map {
                    it.substitution[X]
                }.map {
                    val a = it?.castToList()?.toList()?.map { x -> Clause.of(x.castToStruct()) } ?: emptyList()
                    ClassicSolverFactory.mutableSolverWithDefaultBuiltins(
                        otherLibraries = arg2p.to2pLibraries().plus(FlagsBuilder(
                            argumentLabellingMode = "grounded_hash", graphExtensions = emptyList()
                        ).create().content()),
                        staticKb = Theory.parse(theory + "\n" + creationRules, arg2p.operators()).plus(Theory.of(a)),
                    )
                }.first()
            }
        }

        Thread {
            solver.solve(
                Struct.parse("buildLabelSetsSilent"),
                SolveOptions.allLazilyWithTimeout(TimeDuration.MAX_VALUE)
            ).first()

            update(solver)
            lastSolver = solver
            generationTime = (System.currentTimeMillis() / 1000) - start
        }.also {
            it.start()
            if (blocking) it.join()
        }
    }

    fun launchAutoML(blocking: Boolean, update: (AutoMLResults) -> Unit) {
        lastSolver?.also { solver ->
            Thread {

                println("Saving Graph")
                dataManager.saveKnowledgeBase(nextIteration(), this.theory)
                dataManager.saveGraphData(nextIteration(), dumpGraphData() ?: "")

                println("Exporting AutoML input")
                dataManager.saveAutoMLData(nextIteration(), generationTime, solver)

                println("Input created for iteration ${nextIteration()}")

                execAutoML(dataManager.workspacePath, nextIteration())

                if (dataManager.existsAutoMLData(nextIteration())) {
                    updateIteration()
                    val res = dataManager.loadAutoMLData(config.copy())!!
                    dataManager.saveGeneratedRules(config.copy(),
                        res.inferredRules.distinctBy { it.theoryRepresentation }
                            .joinToString("\n") { "cc${Random.nextLong(0, Long.MAX_VALUE)} :=> ${it.theoryRepresentation}. % ${it.source}" }
                    )
                    update(res)
                } else {
                    // input.delete()
                }
            }.also {
                it.start()
                if (blocking) { it.join() }
            }
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