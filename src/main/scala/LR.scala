import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{SaveMode, SparkSession}

object LR {

  /**
    *
    * <train-data> <preprocess-pipeline-model-path> <test-data> <out-path>
    * @param args
    */

  def main(args:Array[String]) : Unit = {

    if (args.length != 3) {
      println("dude, i need 4 parameters")
    }

    val train_data = args(0)
    val model_path = args(1)
    val test_data = args(2)
    val output = args(3)

    val spark = SparkSession
      .builder()
      .appName("LR")
      //.master("local")
      .getOrCreate()

    val sc = spark.sparkContext
    sc.setLogLevel("ERROR")

    /**
      * Step 1: Load Transformed Training Data
      */
    val train_features_df = spark.read
      .option("header","true")
      .parquet(train_data)
      .repartition(500)

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
      .join(df1.groupBy("app", "day", "hour", "device").agg(count("*") as "app_day_h_dev"), Seq("app", "day", "hour", "device"))
      .repartition(500)
    test_df.show(10)

    val test_features_df = trained_model.transform(test_df).select("label", "features")

    val lr = new LogisticRegression()
      .setMaxIter(10)
//      .setRegParam(0.3)
//      .setElasticNetParam(0.8)

    val paramGrid = new ParamGridBuilder()
      .addGrid(lr.regParam, Array(0.1, 0.01))
      .addGrid(lr.elasticNetParam , Array(0, 0.8))
      .build()

    val tv = new TrainValidationSplit()
      .setEstimator(lr)
      .setEvaluator(new BinaryClassificationEvaluator().setLabelCol("label").setRawPredictionCol("prediction"))
      .setEstimatorParamMaps(paramGrid)
      .setTrainRatio(0.8)  // Use 3+ in practice

    val rfModelWithDF = tv.fit(train_features_df)
//    val rfModelWithDF = gbt.fit(train_features_df)

    val predictions = rfModelWithDF.transform(test_features_df)

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
