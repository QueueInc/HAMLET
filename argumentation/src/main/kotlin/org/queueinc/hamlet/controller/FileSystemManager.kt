package org.queueinc.hamlet.controller

import com.google.gson.Gson
import com.google.gson.JsonObject
import com.google.gson.reflect.TypeToken
import it.unibo.tuprolog.solve.MutableSolver
import org.queueinc.hamlet.createAndWrite
import java.io.File


class FileSystemManager(val workspacePath: String) {

    fun loadConfig() : Config? {
        val configReference = File("$workspacePath/config.json")
        if (configReference.exists()) {
            return Gson().fromJson(configReference.readText(), Config::class.java)
        }

        File("$workspacePath/argumentation").mkdirs()
        File("${workspacePath}/automl/input").mkdirs()
        File("${workspacePath}/automl/output").mkdirs()

        return null
    }

    fun saveConfig(config: Config) {
        File("$workspacePath/config.json").createAndWrite(Gson().toJson(config))
    }

    fun loadKnowledgeBase(config: Config) : String? =
        File("$workspacePath/argumentation/kb_${config.iteration}.txt").let {
            if (it.exists()) it.readText() else null
        }

    fun saveKnowledgeBase(config: Config, theory : String) =
        File("${workspacePath}/argumentation/kb_${config.iteration + 1}.txt").createAndWrite(theory)


    fun loadAutoMLData(config: Config) : AutoMLResults? {
        val outputJson = File("${workspacePath}/automl/output/automl_output_${config.iteration}.json").let { file ->
            if (!file.exists()) return null
            Gson().fromJson(file.readText().trim('\n'), JsonObject::class.java).let {
                val typeToken = object : TypeToken<List<Rule>>() {}.type
                Gson().fromJson<List<Rule>>(it.getAsJsonArray("rules"), typeToken)
            }
        }

        val outputCsv = File("${workspacePath}/automl/output/automl_output_${config.iteration}.csv").let { file ->
            if (!file.exists()) return null
            file.readText().trim('\n').split("\n")
                .mapIndexed { i, el ->
                    listOf(if (i == 0) "iteration" else i.toString()) + el.split(",")
                        .map { it.trim() } }
        }

        return AutoMLResults(outputCsv, outputJson)
    }


    fun saveAutoMLData(config: Config, solver: MutableSolver) {
        val (space, templates, instances) = SpaceTranslator.mineData(solver)
        val (pointsToEvaluate, evaluatedRewards) = loadAutoMLPoints(Config(config.iteration - 1))
        val input = """{
                    "space":$space,
                    "template_constraints":$templates,
                    "instance_constraints":$instances,
                    "points_to_evaluate":$pointsToEvaluate,
                    "evaluated_rewards":$evaluatedRewards
               }""".trimIndent()

        File("${workspacePath}/automl/input/automl_input_${config.iteration}.json").createAndWrite(input)
    }

    fun existsAutoMLData(config: Config) : Boolean =
        File("${workspacePath}/automl/output/automl_output_${config.iteration}.json").exists()

    fun loadGraphData(config: Config) : String? =
        File("${workspacePath}/argumentation/graph_${config.iteration}.txt").let {
            if (!it.exists()) return null
            it.readText()
        }

    fun saveGraphData(config: Config, data: String) =
        File("${workspacePath}/argumentation/graph_${config.iteration}.txt").createAndWrite(data)


    private fun loadAutoMLPoints(config: Config) : Pair<String, String> {
        val output = File("${workspacePath}/automl/output/automl_output_${config.iteration}.json").let {
            if (!it.exists()) return "[]" to "[]"
            it.readText().trim('\n')
        }

        return Gson().fromJson(output, JsonObject::class.java).let {
            it.getAsJsonArray("points_to_evaluate").toString() to
                    it.getAsJsonArray("evaluated_rewards").toString()
        }
    }
}
