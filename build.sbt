name := "ML"

version := "0.1"

scalaVersion := "2.11.8"

val xgboostVersion = "0.72"

val sparkVersion = "2.2.1"


resolvers ++= Seq(
  "apache-snapshots" at "http://repository.apache.org/snapshots/",
  "Local Maven" at Path.userHome.asFile.toURI.toURL + ".m2/repository"
)

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % sparkVersion,
  "org.apache.spark" %% "spark-sql" % sparkVersion,
  "org.apache.spark" %% "spark-mllib" % sparkVersion,
  "org.apache.spark" %% "spark-streaming" % sparkVersion,
  "org.apache.spark" %% "spark-hive" % sparkVersion,
  "ml.dmlc" % "xgboost4j" % xgboostVersion,
  "ml.dmlc" % "xgboost4j-spark" % xgboostVersion

)
