package org.queueinc.hamlet.controller

import com.google.gson.Gson
import com.google.gson.JsonObject
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
import org.queueinc.hamlet.automl.execAutoML
import org.queueinc.hamlet.automl.runAutoML
import org.queueinc.hamlet.automl.stopAutoML
import org.queueinc.hamlet.createAndWrite
import org.queueinc.hamlet.gui.AutoMLResults
import java.io.File


val dataset = "wine"
val metric = "accuracy"
val mode = "max"
val batchSize = 25
val seed = 42

class Controller(private val workspacePath: String, private val dockerMode: Boolean) {

    private val arg2p = Arg2pSolver.default(staticLibs = listOf(SpaceGenerator), dynamicLibs = listOf(SpaceMining))
    private var lastSolver : MutableSolver? = null
    private lateinit var configReference: File

    private var config: Config = Config(0)

    fun init() {

        if (dockerMode) {
            stopAutoML()
            runAutoML(workspacePath)
        }

        configReference = File("$workspacePath/config.json")
        if (configReference.exists()) {
            config = Gson().fromJson(configReference.readText(), Config::class.java)
            return
        }

        File("$workspacePath/argumentation").mkdirs()
        File("${workspacePath}/automl/input").mkdirs()
        File("${workspacePath}/automl/output").mkdirs()
    }

    fun stop() {
        if (dockerMode) {
            stopAutoML()
        }
    }

    fun generateGraph(theory: String, update: (MutableSolver) -> Unit) {

        val solver = ClassicSolverFactory.mutableSolverWithDefaultBuiltins(
            otherLibraries = arg2p.to2pLibraries().plus(FlagsBuilder(
                argumentLabellingMode = "grounded_hash",
                graphExtensions = emptyList()).create().content()),
            staticKb = Theory.parse(theory + "\n" + SpaceGenerator.createGeneratorRules(theory), arg2p.operators())
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

                val input = File("${workspacePath}/automl/input/automl_input_${config.iteration + 1}.json")
                input.createAndWrite(createAutoMLInput(solver))

                println("input created")

                execAutoML(config.iteration + 1, dockerMode)

                if (File("${workspacePath}/automl/input/automl_output_${config.iteration + 1}.json").exists()) {
                    config.iteration++
                    File("${workspacePath}/argumentation/kb_${config.iteration}.txt").createAndWrite(theory)
                    configReference.createAndWrite(Gson().toJson(config))
                    saveGraphData()
                    update(loadAutoMLData()!!)
                } else {
                    input.delete()
                }
            }.start()

        }
    }

    fun loadKnowledgeBase() : String? =
        File("$workspacePath/argumentation/kb_${config.iteration}.txt").let {
            if (it.exists()) it.readText() else null
        }

    fun loadAutoMLData() : AutoMLResults? {
        val output = File("${workspacePath}/automl/output/automl_output_${config.iteration}.csv").let {
            if (!it.exists()) return null
            it.readText().trim('\n')
        }

        return AutoMLResults(
            output.split("\n")
                .mapIndexed { i, el ->
                    listOf(if (i == 0) "iteration" else i.toString()) + el.split(",")
                        .map { it.trim() } })
    }

    fun loadGraphData() : MutableSolver? {
        val graph = File("${workspacePath}/argumentation/graph_${config.iteration}.txt").let {
            if (!it.exists()) return null
            it.readText()
        }

        lastSolver = ClassicSolverFactory.mutableSolverWithDefaultBuiltins(
            otherLibraries = arg2p.to2pLibraries().plus(FlagsBuilder(graphExtensions = emptyList()).create().content()),
        )

        arg2pScope {
            lastSolver!!.solve("miner" call "load_graph_data"(Struct.parse(graph))).first()
        }

        return lastSolver
    }

    private fun createAutoMLInput(solver: MutableSolver) : String {
        val (space, templates, instances) = SpaceTranslator.mineData(solver)
        val (pointsToEvaluate, evaluatedRewards) = loadAutoMLPoints()
        return """{
                    "space":$space,
                    "template_constraints":$templates,
                    "instance_constraints":$instances,
                    "points_to_evaluate":$pointsToEvaluate,
                    "evaluated_rewards":$evaluatedRewards
               }""".trimIndent()
    }

    private fun loadAutoMLPoints() : Pair<String, String> {
        val output = File("${workspacePath}/automl/output/automl_output_${config.iteration}.json").let {
            if (!it.exists()) return "[]" to "[]"
            it.readText().trim('\n')
        }

        return Gson().fromJson(output, JsonObject::class.java).let {
            it.getAsJsonArray("points_to_evaluate").toString() to
                    it.getAsJsonArray("evaluated_rewards").toString()
        }
    }

    private fun saveGraphData() {
        arg2pScope {
            lastSolver?.solve("miner" call "dump_graph_data"(X))
                ?.filter { it.isYes }
                ?.map { it.substitution[X].toString() }
                ?.first()
                ?.also {
                    File("${workspacePath}/argumentation/graph_${config.iteration}.txt").createAndWrite(it)
                }
        }
    }
}