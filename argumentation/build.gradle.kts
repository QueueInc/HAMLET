import com.github.jengelman.gradle.plugins.shadow.tasks.ShadowJar


plugins {
    application
    kotlin("jvm") version "1.5.31"
    id("org.openjfx.javafxplugin") version "0.0.10"
    id("com.github.johnrengelman.shadow") version "7.0.0"
}

group = "org.queueinc"
version = "1.0-SNAPSHOT"

repositories {
    mavenCentral()
}

dependencies {
    implementation(kotlin("stdlib"))
    testImplementation("org.junit.jupiter:junit-jupiter-api:5.6.0")
    testRuntimeOnly("org.junit.jupiter:junit-jupiter-engine")
    implementation("it.unibo.tuprolog.argumentation:arg2p-jvm:0.6.7")
    implementation("it.unibo.tuprolog:solve-classic-jvm:0.20.1")
    implementation("it.unibo.tuprolog:parser-theory-jvm:0.20.1")

    runtimeOnly("org.openjfx:javafx-graphics:16:win")
    runtimeOnly("org.openjfx:javafx-graphics:16:linux")
    runtimeOnly("org.openjfx:javafx-graphics:16:mac")

    implementation("com.google.code.gson:gson:2.9.0")

    /* JUNG DEPENDENCIES */
    api("ch.qos.logback", "logback-classic", "1.2.3")
    api("ch.qos.logback", "logback-core", "1.2.3")
    api("net.sf.jung", "jung-api", "2.1.1")
    api("net.sf.jung", "jung-visualization", "2.1.1")
    api("net.sf.jung", "jung-graph-impl", "2.1.1")
    api("net.sf.jung", "jung-algorithms", "2.1.1")
    api("net.sf.jung", "jung-io", "2.1.1")
}

javafx {
    version = "16"
    modules = listOf("javafx.controls", "javafx.fxml", "javafx.graphics", "javafx.swing")
}

val entryPoint = "org.queueinc.hamlet.Main"

application {
    mainClassName = entryPoint
}

tasks.getByName<Test>("test") {
    useJUnitPlatform()
}
