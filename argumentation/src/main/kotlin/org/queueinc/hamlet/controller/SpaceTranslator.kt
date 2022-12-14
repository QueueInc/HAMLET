package org.queueinc.hamlet.controller

import it.unibo.tuprolog.argumentation.core.dsl.arg2pScope
import it.unibo.tuprolog.core.Struct
import it.unibo.tuprolog.core.Term
import it.unibo.tuprolog.core.parsing.parse
import it.unibo.tuprolog.dsl.prolog
import it.unibo.tuprolog.solve.MutableSolver
import it.unibo.tuprolog.solve.SolveOptions
import it.unibo.tuprolog.solve.TimeDuration
import it.unibo.tuprolog.unify.Unificator
import org.queueinc.hamlet.toSklearnClass
import java.math.BigDecimal

object SpaceTranslator {

    private fun translateList(terms: List<Term>) =
        terms.map {
            when {
                it.isReal -> it.castToReal().value.toDouble().toBigDecimal().toPlainString()
                it.isTrue || it.isNumber || it.isFail -> it
                else -> "\"$it\"".replace("'", "")
            }
    }.toString()

    @JvmStatic
    private fun translateSpace(space: Term) =
        prolog {
            space.castToList().toList().joinToString(",\n") { step ->
                Unificator.default.mgu(step, tupleOf(X, "choice", Z)).let { unifier ->
                    unifier[Z]!!.castToList().toList().map { operator ->
                        if (operator.isAtom) {
                            if (operator.toString() == "function_transformer") {
                                """
                                    {
                                        "type" : "${operator.toSklearnClass()}"
                                    }
                                    """
                            } else "\"$operator\""
                        } else {
                            Unificator.default.mgu(operator, tupleOf(A, B)).let { unifier2 ->
                                unifier2[B]!!.castToList().toList().joinToString(",\n") { hyper ->
                                    Unificator.default.mgu(hyper, tupleOf(C, D, E)).let { unifier3 ->
                                        """
                                            "${unifier3[C].toString().replace("'", "")}" : {
                                               "${unifier3[D]}" : ${translateList(unifier3[E]!!.castToList().toList())}
                                            }
                                            """
                                    }
                                }.let {
                                    """
                                        {
                                            "type" : "${unifier2[A]!!.toSklearnClass()}"${if (it.isEmpty()) "" else ","}
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
                    {
                        $it
                    }
                    """.replace("\\s".toRegex(), "")
            }
        }

    @JvmStatic
    private fun translateTemplates(mandatory: Term, forbidden: Term, mandatoryOrder: Term) : String {

        val mapTerm = { step: Term, target: List<Term> ->
            if (step.toString() == "prototype") target.map { "\"${it.castToList().toList().joinToString("_")}\"" }
            else target.map { "\"${it.toSklearnClass()}\"" }
        }

        val transform = { target: Term, comparator: String, type: Boolean ->
            prolog {
                target.castToList().toList().map { template ->
                    Unificator.default.mgu(template, tupleOf(A, B, C)).let { unifier ->
                        unifier[A]!!.castToList().toList().mapIndexed { i, step ->
                            if (type || i == 0) {
                                """
                                    "$step" : {${if (type) "\"type\": {" else ""}"$comparator": ${mapTerm(step, unifier[B]!!.castToList().toList()[i].castToList().toList())} ${if (type) "}" else ""}}
                                """
                            }
                            else {
                                """
                                    "$step" : { "type": { "neq": "${Term.parse("function_transformer").toSklearnClass()}" }}
                                """
                            }
                        }.joinToString(",\n").let {
                            """
                                {
                                    ${it + if (it == "") "" else ","}
                                    "classification": {"type": {"eq": "${unifier[C]!!.toSklearnClass()}"}}
                                }
                                """
                        }
                    }
                }
            }
        }
        return """
                ${transform(mandatory, "nin", true) + transform(forbidden, "in", true) + transform(mandatoryOrder, "nin", false)}
            """.replace("\\s".toRegex(), "")
    }

    @JvmStatic
    private fun translateInstances(instances: Term) =
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
                                    "\"${hyper[G]}\" : ${if (hyper[H]!!.isTrue || hyper[H]!!.isNumber || hyper[H]!!.isFail) hyper[H] else "\"${hyper[H]}\"".replace("'", "")}"
                                }
                            }.let { hyperparameters ->
                                """
                                    "${it[D]}": {
                                        "type": "${it[E]!!.toSklearnClass()}"${if (hyperparameters.isEmpty()) "" else ","}
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
            }.toString().replace("\\s".toRegex(), "")
        }

    @JvmStatic
    fun mineData(solver: MutableSolver) =
        arg2pScope {

            println("Exporting Space")

            val space = solver.solve("miner" call "fetch_complete_space"(X), SolveOptions.allLazilyWithTimeout(TimeDuration.MAX_VALUE))
                .filter { it.isYes }
                .map { translateSpace(it.substitution[X]!!) }
                .first()

            println("Exporting Constrainst")

            val templates = solver.solve(("miner" call "fetch_mandatory"(X)) and ("miner" call "fetch_forbidden"(Y)) and ("miner" call "fetch_mandatory_order"(Z)), SolveOptions.allLazilyWithTimeout(TimeDuration.MAX_VALUE))
                .filter { it.isYes }
                .map { translateTemplates(it.substitution[X]!!, it.substitution[Y]!!, it.substitution[Z]!!) }
                .first()

            println("Exporting Instances")

            val instances = solver.solve("miner" call "fetch_instance_base_components"(X, Y), SolveOptions.allLazilyWithTimeout(TimeDuration.MAX_VALUE))
                .filter { it.isYes }
                .map {
                    println("Collected Raw Instances")
                    instances(it.substitution[X]!!, it.substitution[Y]!!)
                }
                .first()

            arrayOf(space, templates, instances)
        }

    @JvmStatic
    fun instances(operators: Term, pipelines: Term) : String {
        val operatorList = operators.castToList().toList()
        val pipelinesList = pipelines.castToList().toList().map { it.castToList().toList() }

        val checkStep = { operator: Term, step: Term ->
            operator.toString().contains(step.toString())
        }

        val checkPipeline = { pipeline: Term ->
            pipeline.toString().let {
                it.contains("function_transformer") ||
                        it.contains("classification") ||
                        it.contains("prototype")
            }
        }

        val mapPrototype = { pipeline: List<Term> ->
            pipeline.joinToString("_") { it.toString().substringBefore(",").replace("(", "").trim() }
        }

        val instances = pipelinesList.flatMap { pipeline ->
            pipeline.filter { step -> !checkPipeline(step) }.map { step ->
                operatorList.filter { operator -> checkStep(operator, step)}
            }.fold(listOf(pipeline)) { acc, set ->
                acc.flatMap { pipe -> set.map { op -> pipe.map {
                    if (checkStep(op, it)) op else it
                } } }
            }.map { elem ->
                it.unibo.tuprolog.core.List.of(elem + listOf(Struct.parse("(prototype, ${mapPrototype(elem)})")))
            }
        }.let { elem ->
            it.unibo.tuprolog.core.List.of(elem)
        }

        return translateInstances(instances)
    }
}