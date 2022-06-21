package org.queueinc.hamlet.argumentation

import it.unibo.tuprolog.argumentation.core.Arg2pSolver
import it.unibo.tuprolog.argumentation.core.libs.ArgLibrary
import it.unibo.tuprolog.argumentation.core.libs.ArgsFlag
import it.unibo.tuprolog.argumentation.core.libs.utils.*
import it.unibo.tuprolog.core.Atom
import it.unibo.tuprolog.core.Integer
import it.unibo.tuprolog.core.Substitution
import it.unibo.tuprolog.core.Term
import it.unibo.tuprolog.solve.ExecutionContext
import it.unibo.tuprolog.solve.Signature
import it.unibo.tuprolog.solve.exception.error.TypeError
import it.unibo.tuprolog.solve.library.AliasedLibrary
import it.unibo.tuprolog.solve.library.Library
import it.unibo.tuprolog.solve.primitive.Primitive
import it.unibo.tuprolog.solve.primitive.Solve
import it.unibo.tuprolog.theory.Theory
import it.unibo.tuprolog.theory.parsing.parse
import java.util.*
import kotlin.collections.ArrayList
import kotlin.random.Random


object SpaceGenerator : ArgLibrary {

    override val alias = "hamlet.core"

    override val baseContent: AliasedLibrary
        get() = Library.aliased(
            alias = alias,
            theory = Theory.parse(
                SpaceGenerator::class.java.getResource("engine.pl").let {
                    it!!.readText()
                }, Arg2pSolver.default().operators()
            ),
            primitives = mapOf(
                Rand.signature to Rand,
            )
        )

    override val baseFlags: Iterable<ArgsFlag<*, *>>
        get() = emptyList()


    fun createGeneratorRules(theory: String) : String {

        val getStepNames = { stepNames: Int ->
            IntRange(1, stepNames).joinToString(",") { "Step_$it" }
        }
        val getSteps = { steps: Int ->
            IntRange(1, steps).joinToString(", ") { "step(Step_$it)" }
        }
        val getStepsCheck = { steps: Int ->
            IntRange(1, steps).joinToString(", ") { "Step_$it \\= classification" }
        }
        val getStepsComparison = { steps: Int ->
            if (steps <= 1) ""
            else ", " + IntRange(1, steps).flatMap { i -> IntRange(i + 1, steps).map { "Step_$i \\= Step_$it" } }.joinToString(", ")
        }

        val regex = """step\((\w+)\)\.""".toRegex()
        val matchResult = regex.findAll(theory).map { it.groupValues[1] }.filterNot { it == "classification" }.toList()
        val matchResultCount = matchResult.count()

        return IntRange(1, matchResultCount).map { i ->
            """pipeline([${getStepNames(i)}], ZZ) :- ${getSteps(i)}, ${getStepsCheck(i)} ${getStepsComparison(i)}, operator(classification, ZZ)."""
                .trimIndent()
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

object Rand : Primitive {

    val signature: Signature = Signature("rand_int", 3)

    override fun solve(request: Solve.Request<ExecutionContext>): Sequence<Solve.Response> {
        val start: Term = request.arguments[0]
        val end: Term = request.arguments[1]
        val res: Term = request.arguments[2]

        if (start !is Integer) {
            throw TypeError.forGoal(
                request.context,
                request.signature,
                TypeError.Expected.INTEGER,
                start
            )
        }

        if (end !is Integer) {
            throw TypeError.forGoal(
                request.context,
                request.signature,
                TypeError.Expected.INTEGER,
                end
            )
        }

        if (res !is it.unibo.tuprolog.core.Var) {
            throw TypeError.forGoal(
                request.context,
                request.signature,
                TypeError.Expected.VARIABLE,
                res
            )
        }

        return sequence {
            yield(
                request.replyWith(Substitution.of(res, Atom.of("p" + Random.nextInt(start.intValue.toInt(), end.intValue.toInt()))))
            )
        }
    }
}