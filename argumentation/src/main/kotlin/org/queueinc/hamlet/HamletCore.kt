package org.queueinc.hamlet

import it.unibo.tuprolog.argumentation.core.Arg2pSolver
import it.unibo.tuprolog.argumentation.core.libs.ArgLibrary
import it.unibo.tuprolog.argumentation.core.libs.ArgsFlag
import it.unibo.tuprolog.solve.library.AliasedLibrary
import it.unibo.tuprolog.solve.library.Library
import it.unibo.tuprolog.theory.Theory
import it.unibo.tuprolog.theory.parsing.parse

object HamletCore : ArgLibrary {

    override val alias = "hamlet.core"

    override val baseContent: AliasedLibrary
        get() = Library.aliased(
            alias = this.alias,
            theory = Theory.parse(
                HamletCore::class.java.getResource("engine.pl").let {
                    it!!.readText()
                }, Arg2pSolver.default().operators()
            )
        )

    override val baseFlags: Iterable<ArgsFlag<*, *>>
        get() = emptyList()

}