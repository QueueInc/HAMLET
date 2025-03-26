package org.queueinc.hamlet.argumentation

import it.unibo.tuprolog.argumentation.core.Arg2pSolver
import it.unibo.tuprolog.argumentation.core.libs.ArgLibrary
import it.unibo.tuprolog.argumentation.core.libs.ArgsFlag
import it.unibo.tuprolog.argumentation.core.libs.Loadable
import it.unibo.tuprolog.solve.library.Library
import it.unibo.tuprolog.theory.Theory
import it.unibo.tuprolog.theory.parsing.parse

object SpaceMining : ArgLibrary, Loadable {

    override val alias = "hamlet.mining"

    override val baseContent: Library
        get() = Library.of(
            alias = alias,
            clauses = Theory.parse(
                SpaceMining::class.java.getResource("graph_mining.pl").let {
                    it!!.readText()
                }, Arg2pSolver.default().operators()
            )
        )

    override val baseFlags: Iterable<ArgsFlag<*, *>>
        get() = emptyList()

    override fun identifier(): String = "miner"
}

