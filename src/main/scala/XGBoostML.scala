
import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.ml.evaluation.{BinaryClassificationEvaluator, MulticlassClassificationEvaluator, RegressionEvaluator}
import org.apache.spark.sql.{SaveMode, SparkSession}
import org.apache.spark.sql.functions._

import scala.collection.mutable.ArrayBuffer
import org.apache.spark.ml.feature.{OneHotEncoder, StringIndexer}
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.{Pipeline, PipelineModel}
import java.util.Calendar

import org.apache.log4j.{Level, Logger}
import ml.dmlc.xgboost4j.scala.spark.{XGBoost, XGBoostEstimator}
import org.apache.spark.ml.feature._
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}
import org.apache.spark.sql._
import org.apache.spark.sql.functions.lit

object XGBoostML {

  /**
    *
    * <train-data> <preprocess-pipeline-model-path> <test-data> <out-path>
    * @param args
    */

  def main(args:Array[String]) : Unit = {

    if (args.length != 4) {
      println("dude, i need 4 parameters")
    }

    val train_data = args(0)
    val model_path = args(1)
    val test_data = args(2)
    val output = args(3)


    val spark = SparkSession
      .builder()
      .appName("XGB")
      .getOrCreate()

    val sc = spark.sparkContext
    sc.setLogLevel("ERROR")

    //    import spark.sqlContext.implicits._


    /**
      * Step 1: Load Transformed Training Data
      */
    val train_features_df = spark.read
      .option("header","true")
      .parquet(train_data)
      .repartition(4)

    train_features_df.cache()


    val trained_model = PipelineModel.load(model_path)

    val df1 = spark.read
      .option("header","true")
      .option("inferSchema","true")
      .csv(test_data)
      .withColumn("hour",hour(col("click_time")))
      .withColumn("day",dayofmonth(col("click_time")))
      .drop("click_time","attributed_time")
      .withColumnRenamed("is_attributed", "label")

    val test_df = df1
      .join(df1.groupBy("ip", "hour", "device").agg(count("*") as "ip_h_dev"), Seq("ip", "hour", "device"))
      .join(df1.groupBy("ip", "hour", "app").agg(count("*") as "ip_h_app"), Seq("ip", "hour", "app"))
      .join(df1.groupBy("ip", "hour", "os").agg(count("*") as "ip_h_os"), Seq("ip", "hour", "os"))
      .join(df1.groupBy("ip", "day", "hour").agg(count("*") as "ip_day_h"), Seq("ip", "day", "hour"))
      .join(df1.groupBy("ip", "os", "device").agg(count("*") as "ip_os_dev"), Seq("ip", "os", "device"))
      .join(df1.groupBy("ip", "day", "hour", "app").agg(count("*") as "nipApp"), Seq("ip", "day", "hour", "app"))
      //      .join(df1.groupBy("ip", "day", "hour", "app", "os").agg(count("*") as "nipAppOs"), Seq("ip", "day", "hour", "app", "os"))
      .join(df1.groupBy("app", "day", "hour", "device").agg(count("*") as "app_day_h_dev"), Seq("app", "day", "hour", "device"))
    test_df.show(10)

    val test_features_df = trained_model.transform(test_df).select("label", "features")

    // number of iterations
    val numRound = 10
    val numWorkers = 4
    // training parameters
    val paramMap = List(
      "nWorkers" -> 2,
      "eta" -> 0.1,
      "max_depth" -> 6,
      "min_child_weight" -> 3.0,
      "subsample" -> 0.8,
      "colsample_bytree" -> 0.82,
      "colsample_bylevel" -> 0.9,
      "base_score" -> 0.005,
      "eval_metric" -> "auc",
      "seed" -> 49,
      "silent" -> 1,
      "objective" -> "binary:logistic").toMap
    println("Starting Xgboost ")


    val xgbEstimator = new XGBoostEstimator(paramMap).setFeaturesCol("features").
      setLabelCol("label")
    val paramGrid = new ParamGridBuilder()
      .addGrid(xgbEstimator.round, Array(20, 50))
      .addGrid(xgbEstimator.eta, Array(0.1, 0.4))
      .build()
    val tv = new TrainValidationSplit()
      .setEstimator(xgbEstimator)
      .setEvaluator(new BinaryClassificationEvaluator().setLabelCol("label").setRawPredictionCol("prediction"))
      .setEstimatorParamMaps(paramGrid)
      .setTrainRatio(0.8)  // Use 3+ in practice
    val xgBoostModelWithDF = tv.fit(train_features_df)

//    val xgBoostModelWithDF = XGBoost.trainWithDataFrame(
//      train_features_df,
//      paramMap,
//      round = numRound,
//      nWorkers = numWorkers,
//      useExternalMemory = true)

    val predictions = xgBoostModelWithDF.transform(test_features_df)

    predictions.show(10)

    test_features_df.repartition(50).write
      .mode(SaveMode.Overwrite)
      .save(output + "_tests.parquet")

    predictions.repartition(50).write
      .mode(SaveMode.Overwrite)
      .save(output + "_predicts.parquet")

    test_features_df.repartition(50).printSchema()
    predictions.printSchema()

  }






}
