package org.queueinc.hamlet.argumentation

import it.unibo.tuprolog.argumentation.core.Arg2pSolver
import it.unibo.tuprolog.argumentation.core.libs.ArgLibrary
import it.unibo.tuprolog.argumentation.core.libs.ArgsFlag
import it.unibo.tuprolog.solve.library.AliasedLibrary
import it.unibo.tuprolog.solve.library.Library
import it.unibo.tuprolog.theory.Theory
import it.unibo.tuprolog.theory.parsing.parse

object SpaceGenerator : ArgLibrary {

    override val alias = "hamlet.core"

    override val baseContent: AliasedLibrary
        get() = Library.aliased(
            alias = alias,
            theory = Theory.parse(
                SpaceGenerator::class.java.getResource("engine.pl").let {
                    it!!.readText()
                }, Arg2pSolver.default().operators()
            )
        )

    override val baseFlags: Iterable<ArgsFlag<*, *>>
        get() = emptyList()


    fun createGeneratorRules(theory: String) : String {

        val getStepNames = { number: Int ->
            IntRange(1, number).map { "X".repeat(it) }
        }
        val getSteps = { stepNames: List<String> ->
            stepNames.joinToString(", ") { "step($it)" }
        }
        val checkDistinct = { stepNames: List<String> ->
            if (stepNames.count() <= 1) ""
            else stepNames.flatMapIndexed { i, out -> stepNames.drop(i).filter { it != out }.map { """$it \= $out""" } }
                .joinToString(", ") + ", "
        }
        val checkNoClassification = { stepNames: List<String> ->
            stepNames.joinToString(", ") { """$it \= classification""" }
        }
        val getOperatorNames = { stepNames: List<String> ->
            stepNames.joinToString(",") { "${it}Y" }
        }
        val getOperators = { stepNames: List<String> ->
            stepNames.joinToString(", ") { "operator($it, ${it}Y)" }
        }

        val regex = """step\(\w+\)\.""".toRegex()
        val matchResult = regex.findAll(theory).count()

        return IntRange(1, matchResult).joinToString("\n") {
            val stepNames = getStepNames(it)
            """g$it : ${getSteps(stepNames)}, step(classification), prolog((${checkDistinct(stepNames)} ${checkNoClassification(stepNames)})), 
                    ${getOperators(stepNames)}, operator(classification, ZZ) => pipeline([${getOperatorNames(stepNames)}], ZZ)."""
                .trimIndent()
        }
    }

}