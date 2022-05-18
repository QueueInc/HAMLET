package org.queueinc.hamlet

import javafx.application.Application

object Starter {
    @JvmStatic
    fun main(args: Array<String>) {
        workspacePath = args[0]
        Application.launch(HamletGUI::class.java)
    }
}