import java.time.format.DateTimeFormatter
import java.time.LocalDateTime

val gitCommit: String? by project
val buildTime = LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyyMMddHHmmss"))

version = if (!gitCommit.isNullOrBlank()) {
    "$buildTime-$gitCommit"
} else {
    "$buildTime"
}

plugins {
    id("java")
    id("com.gradleup.shadow") version "9.6.1"
}

group = "com.mc1510ty"
version = "1.0-SNAPSHOT"

repositories {
    mavenCentral()
}

tasks{
    shadowJar {
        archiveFileName.set("TekitouLLM-${project.version}.jar")
        manifest {
            attributes["Main-Class"] = "com.mc1510ty.TekitouLLM.Main"
        }
    }
}



tasks.register<Exec>("run") {
    dependsOn("shadowJar")

    val shadowJarTask = tasks.named<com.github.jengelman.gradle.plugins.shadow.tasks.ShadowJar>("shadowJar")
    val jarFile = shadowJarTask.get().archiveFile.get().asFile.absolutePath

    if (System.getProperty("os.name").lowercase().contains("windows")) {
        // cmd /k に渡すコマンド全体をひとつの文字列にまとめる
        val command = "chcp 65001 && java -Xmx16G -jar \"$jarFile\" && pause && exit"

        // ★ start の直後に "" (空タイトル) を追加します
        commandLine("cmd", "/c", "start", "", "/ABOVENORMAL", "/AFFINITY", "FFF", "cmd", "/k", command)
    } else {
        commandLine("java", "--enable-native-access=ALL-UNNAMED", "--sun-misc-unsafe-memory-access=allow", "-jar", jarFile)
    }
}
