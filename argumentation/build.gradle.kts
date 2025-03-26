import com.github.jengelman.gradle.plugins.shadow.tasks.ShadowJar
import java.io.FileOutputStream
import java.util.Properties

plugins {
    application
    kotlin("jvm") version "1.9.25"
    id("org.openjfx.javafxplugin") version "0.1.0"
    id("com.github.johnrengelman.shadow") version "8.1.1"
}

group = "org.queueinc"
version = "1.0.40-fairness-dev"

repositories {
    mavenCentral()
}


dependencies {
    implementation(kotlin("stdlib"))
    testImplementation("org.junit.jupiter:junit-jupiter-api:5.8.2")
    testRuntimeOnly("org.junit.jupiter:junit-jupiter-engine:5.8.2")
    implementation("it.unibo.tuprolog.argumentation:arg2p-jvm:0.10.2")

    runtimeOnly("org.openjfx:javafx-graphics:19.0.2.1:win")
    runtimeOnly("org.openjfx:javafx-graphics:19.0.2.1:linux")
    runtimeOnly("org.openjfx:javafx-graphics:19.0.2.1:mac")

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
    version = "21"
    modules = listOf("javafx.controls", "javafx.fxml", "javafx.graphics", "javafx.swing")
}

application {
    mainClass = "org.queueinc.hamlet.Main"
}

java {
    toolchain.languageVersion.set(JavaLanguageVersion.of(21))
}

tasks.withType<org.jetbrains.kotlin.gradle.tasks.KotlinCompile> {
    kotlinOptions {
        jvmTarget = "21"
    }
}

tasks.getByName<Test>("test") {
    useJUnitPlatform()
}


val generatedVersionDir = "${layout.buildDirectory}/generated-version"

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
