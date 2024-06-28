import com.github.jengelman.gradle.plugins.shadow.tasks.ShadowJar
import java.io.FileOutputStream
import java.util.Properties

plugins {
    application
    kotlin("jvm") version "1.6.21"
    id("org.openjfx.javafxplugin") version "0.0.10"
    id("com.github.johnrengelman.shadow") version "7.0.0"
}

group = "org.queueinc"
version = "1.0.24-fairness-dev"

repositories {
    mavenCentral()
}


dependencies {
    implementation(kotlin("stdlib"))
    testImplementation("org.junit.jupiter:junit-jupiter-api:5.8.2")
    testRuntimeOnly("org.junit.jupiter:junit-jupiter-engine:5.8.2")
    implementation("it.unibo.tuprolog.argumentation:arg2p-jvm:0.6.8")
    implementation("it.unibo.tuprolog:solve-classic-jvm:0.20.4")
    implementation("it.unibo.tuprolog:parser-theory-jvm:0.20.4")

    runtimeOnly("org.openjfx:javafx-graphics:18.0.1:win")
    runtimeOnly("org.openjfx:javafx-graphics:18.0.1:linux")
    runtimeOnly("org.openjfx:javafx-graphics:18.0.1:mac")

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

tasks.withType<org.jetbrains.kotlin.gradle.tasks.KotlinCompile> {
    kotlinOptions {
        jvmTarget = "11"
    }
}

tasks.getByName<Test>("test") {
    useJUnitPlatform()
}


val generatedVersionDir = "$buildDir/generated-version"

sourceSets {
    main {
        kotlin {
            output.dir(generatedVersionDir)
        }
    }
}

tasks.register("generateVersionProperties") {
    doLast {
        val propertiesFile = file("$generatedVersionDir/version.properties")
        propertiesFile.parentFile.mkdirs()
        val properties = Properties()
        properties.setProperty("version", "$version")
        val out = FileOutputStream(propertiesFile)
        properties.store(out, null)
    }
}

tasks.named("processResources") {
    dependsOn("generateVersionProperties")
}
