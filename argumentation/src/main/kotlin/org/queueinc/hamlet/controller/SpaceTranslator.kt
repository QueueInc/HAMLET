package org.queueinc.hamlet.controller

import it.unibo.tuprolog.argumentation.core.dsl.arg2pScope
import it.unibo.tuprolog.core.Term
import it.unibo.tuprolog.dsl.prolog
import it.unibo.tuprolog.solve.MutableSolver
import it.unibo.tuprolog.solve.SolveOptions
import it.unibo.tuprolog.solve.TimeDuration
import it.unibo.tuprolog.unify.Unificator
import org.queueinc.hamlet.toSklearnClass

object SpaceTranslator {

    @JvmStatic
    private fun translateSpace(space: Term) =
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
    private fun translateTemplates(mandatory: Term, forbidden: Term) : String {
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
                ${transform(mandatory, "nin") + transform(forbidden, "in")}
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
                                    "\"${hyper[G]}\" : ${hyper[H]}"
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

            arrayOf(space, templates, instances)
        }
}