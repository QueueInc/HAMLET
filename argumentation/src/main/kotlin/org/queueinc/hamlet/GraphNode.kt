package org.queueinc.hamlet

import it.unibo.tuprolog.solve.MutableSolver
import javafx.embed.swing.SwingNode

data class GraphNode(val node : SwingNode, val update : (MutableSolver) -> it.unibo.tuprolog.argumentation.core.model.Graph)
