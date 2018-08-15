import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.sql.{SaveMode, SparkSession}
import org.apache.spark.sql.functions._

import scala.collection.mutable.ArrayBuffer
import org.apache.spark.ml.feature.{OneHotEncoder, StringIndexer}
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.{Pipeline, PipelineModel}

object DecisionTree {
  def main(args:Array[String]) : Unit = {

    if (args.length != 3) {
      println("dude, i need three parameters")
    }

    val train_data = args(0)
    val model_path = args(1)
    val test_data = args(2)
    val output = args(3)

    val spark = SparkSession
      .builder()
      .appName("DecisionTree")
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


    val trained_model = PipelineModel.load(model_path)

    val test_df = spark.read
      .option("header","true")
      .option("inferSchema","true")
      .csv(test_data)
      .withColumn("hour",hour(col("click_time")))
      .withColumn("day",dayofmonth(col("click_time")))
      .drop("click_time","attributed_time")
      .withColumnRenamed("is_attributed", "label")

    val test_features_df = trained_model.transform(test_df)

    val dt = new DecisionTreeClassifier()

    // Train model.
    val model = dt.fit(train_features_df)

    // Make Predictions
    val predictions = model.transform(test_features_df)

    // Export prediction for later use.
    val prediction_path = output + "_prediction"
    predictions.write.option("header", "true").csv(prediction_path)

    val vectorized_test_data_path = output + "_vectorized_test_data"
    test_features_df.write.option("header", "true").parquet(vectorized_test_data_path)

    val dt_model_path = output + "_trained_dt_model"
    model.save(dt_model_path)

  }
}




