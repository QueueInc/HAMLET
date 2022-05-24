package org.queueinc.hamlet.argumentation

import it.unibo.tuprolog.argumentation.core.Arg2pSolver
import it.unibo.tuprolog.argumentation.core.libs.ArgLibrary
import it.unibo.tuprolog.argumentation.core.libs.ArgsFlag
import it.unibo.tuprolog.solve.library.AliasedLibrary
import it.unibo.tuprolog.solve.library.Library
import it.unibo.tuprolog.theory.Theory
import it.unibo.tuprolog.theory.parsing.parse
import java.util.*
import kotlin.collections.ArrayList


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

        val getOperatorNames = { stepNames: List<String> ->
            stepNames.joinToString(",") { "Y${it}" }
        }
        val getOperators = { stepNames: List<String> ->
            stepNames.joinToString(", ") { "operator($it, Y${it})" }
        }

        val regex = """step\((\w+)\)\.""".toRegex()
        val matchResult = regex.findAll(theory).map { it.groupValues[1] }.filterNot { it == "classification" }.toList()
        val matchResultCount = matchResult.count()

        return IntRange(1, matchResultCount).flatMap { elements ->
            permK(matchResult, 0, elements).mapIndexed { i, stepNames ->
                """g$elements$i : ${getOperators(stepNames)}, operator(classification, ZZ) => pipeline([${getOperatorNames(stepNames)}], ZZ)."""
                    .trimIndent()
            }
        }.joinToString("\n")
    }

    private fun <E> permK(p: List<E>, i: Int, k: Int) : List<List<E>> {
        if (i == k) {
            return listOf(ArrayList(p.subList(0, k)))
        }
        val r = mutableListOf<List<E>>()
        for (j in i until p.size) {
            Collections.swap(p, i, j)
            r.addAll(permK(p, i + 1, k))
            Collections.swap(p, i, j)
        }
        return r
    }

}