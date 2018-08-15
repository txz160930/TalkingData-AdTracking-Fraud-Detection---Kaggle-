import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.sql.{SaveMode, SparkSession}

object Evaluation {
  def main(args:Array[String]) : Unit = {

    if (args.length != 2) {
      println("dude, i need three parameters")
    }

    val test_predict_data = args(0)
    val output = args(1)

    val spark = SparkSession
      .builder()
      .appName("Evaluation")
      .getOrCreate()

    import spark.implicits._

    val sc = spark.sparkContext


    sc.setLogLevel("ERROR")

    val test_predict_df = spark.read
      .option("header", "true")
      .parquet(test_predict_data)

    test_predict_df.show(10)

    val evaluator = new BinaryClassificationEvaluator()
      .setRawPredictionCol("prediction")
      .setMetricName("areaUnderROC")
    val evaluatorParams = ParamMap(evaluator.metricName -> "areaUnderROC")
    val areaTest = evaluator.evaluate(test_predict_df, evaluatorParams)
    println("Evaluation: areaUnderROC " + areaTest.toString)


    sc.parallelize(Seq(areaTest.toString))
      .toDF
      .write
      .mode(SaveMode.Overwrite)
      .csv(output + "_auroc")

  }

}
