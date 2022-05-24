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
    private var oldKb = ""

    fun init() : String {

        if (dockerMode) {
            stopAutoML()
            runAutoML(workspacePath)
        }

        configReference = File("$workspacePath/config.json")
        if (configReference.exists()) {
            config = Gson().fromJson(configReference.readText(), Config::class.java)
            oldKb = File("$workspacePath/argumentation/kb_${config.iteration}.txt").readText()
        }
        else {
            File("$workspacePath/argumentation").mkdirs()
            File("${workspacePath}/automl/input").mkdirs()
            File("${workspacePath}/automl/output").mkdirs()
        }

        return oldKb
    }

    fun stop() {
        if (dockerMode) {
            stopAutoML()
        }
    }

    fun generateGraph(theory: String, update: (MutableSolver) -> Unit) {

        val generator = SpaceGenerator.createGeneratorRules(theory)
        println(generator)

        val solver = ClassicSolverFactory.mutableSolverWithDefaultBuiltins(
            otherLibraries = arg2p.to2pLibraries().plus(FlagsBuilder(graphExtensions = emptyList()).create().content()),
            staticKb = Theory.parse(theory + "\n" + generator, arg2p.operators())
        )

        Thread {
            solver.solve(
                Struct.parse("buildLabelSets(_, _)"),
                SolveOptions.allLazilyWithTimeout(TimeDuration.MAX_VALUE)).first()
            update(solver)
            lastSolver = solver
        }.start()
    }

    fun launchAutoML(theory: String, update: (AutoMLResults) -> Unit) {
        lastSolver?.also { solver ->

            val (space, templates, instances) = SpaceTranslator.mineData(solver)
            val (pointsToEvaluate, evaluatedRewards) = loadAutoMLPoints()

            config.iteration++
            val input = File("${workspacePath}/automl/input/automl_input_${config.iteration}.json")
            input.createAndWrite("{\"space\":$space,\"template_constraints\":$templates,\"instance_constraints\":$instances,\"points_to_evaluate\":$pointsToEvaluate,\"evaluated_rewards\":$evaluatedRewards}")

            Thread {
                execAutoML(config.iteration, dockerMode)
                if (File("${workspacePath}/automl/input/automl_output_${config.iteration}.json").exists()) {
                    File("${workspacePath}/argumentation/kb_${config.iteration}.txt").createAndWrite(theory)
                    configReference.createAndWrite(Gson().toJson(config))
                    saveGraphData()
                    update(loadAutoMLData()!!)
                } else {
                    // input.delete()
                }
            }.start()

        }
    }

    fun loadAutoMLPoints() : Pair<String, String> {
        val output = File("${workspacePath}/automl/output/automl_output_${config.iteration}.json").let {
            if (!it.exists()) return "[]" to "[]"
            it.readText().trim('\n')
        }

        return Gson().fromJson(output, JsonObject::class.java).let {
            it.getAsJsonArray("points_to_evaluate").toString() to
                    it.getAsJsonArray("evaluated_rewards").toString()
        }
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

    fun saveGraphData() {
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
}