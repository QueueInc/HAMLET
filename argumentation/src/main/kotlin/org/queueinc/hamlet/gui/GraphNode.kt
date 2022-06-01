package org.queueinc.hamlet.gui

import it.unibo.tuprolog.solve.MutableSolver
import javafx.embed.swing.SwingNode

data class GraphNode(val node : SwingNode, val update : (MutableSolver) -> Unit)
