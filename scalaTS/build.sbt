name := "ScalaTS"

version := "1.0.0"

organization := "com.suning"

scalaVersion := "2.11.8"

libraryDependencies ++= Seq(
  "com.typesafe.scala-logging" %% "scala-logging-slf4j"   % "2.1.2",
  "org.apache.spark"           %% "spark-core"            % "2.0.0"   % "provided",
  "org.apache.spark"           %% "spark-mllib-local"     % "2.0.0"   % "provided",
  "org.apache.spark"           %% "spark-mllib"           % "2.0.0"   % "provided",
  "org.apache.spark"           %% "spark-sql"             % "2.0.0"   % "provided",
  "org.scalatest"              %% "scalatest"             % "3.0.0",
  "org.scalaz"                 %% "scalaz-core"           % "7.2.6",
  "org.apache.commons"          % "commons-lang3"         % "3.4",
  "org.scalanlp"               %% "breeze"                % "0.11",
  "org.scalanlp"               %% "breeze-natives"        % "0.11",
  "org.scalanlp"               %% "breeze-viz"            % "0.11",
  "org.apache.spark"           %% "spark-hive"            % "2.0.0",
  "org.ansj"                    % "ansj_seg"              % "5.0.3"
)

test in assembly := {}

parallelExecution in Test := false