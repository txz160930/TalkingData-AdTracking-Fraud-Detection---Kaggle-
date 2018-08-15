import org.apache.spark.sql.{SaveMode, SparkSession}
import org.apache.spark.sql.functions._
import scala.collection.mutable.ArrayBuffer
import org.apache.spark.ml.feature.{StringIndexer,OneHotEncoder}
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.{Pipeline, PipelineModel}

object Preprocess {
  def main(args:Array[String]) : Unit = {

    if (args.length != 2) {
      println("dude, i need two parameters")
    }

    val input = args(0)
    val output = args(1)

    val spark = SparkSession
      .builder()
      .appName("Preprocess")
      .getOrCreate()

    val sc = spark.sparkContext
    sc.setLogLevel("ERROR")

    val filePath = input
    val trainData = spark.read.option("header","true").option("inferSchema","true").csv(filePath)
    trainData.printSchema

    val Array(train, test) = trainData.randomSplit(Array(0.9, 0.1), seed = 0)

    // Split test data and save to disk
    test
      .write
      .option("header","true")
      .mode(SaveMode.Overwrite)
      .csv(output + "_test")

    val df1 = train//.withColumn("minute", minute(col("click_time")))
      .withColumn("hour",hour(col("click_time")))
      .withColumn("day",dayofmonth(col("click_time")))
      .drop("click_time","attributed_time")
      .withColumnRenamed("is_attributed", "label")

    df1.show(10)

    val trainDF = df1
      .join(df1.groupBy("ip", "hour", "device").agg(count("*") as "ip_h_dev"), Seq("ip", "hour", "device"))
      .join(df1.groupBy("ip", "hour", "app").agg(count("*") as "ip_h_app"), Seq("ip", "hour", "app"))
      .join(df1.groupBy("ip", "hour", "os").agg(count("*") as "ip_h_os"), Seq("ip", "hour", "os"))
      .join(df1.groupBy("ip", "day", "hour").agg(count("*") as "ip_day_h"), Seq("ip", "day", "hour"))
      .join(df1.groupBy("ip", "os", "device").agg(count("*") as "ip_os_dev"), Seq("ip", "os", "device"))
      .join(df1.groupBy("ip", "day", "hour", "app").agg(count("*") as "nipApp"), Seq("ip", "day", "hour", "app"))
      .join(df1.groupBy("app", "day", "hour", "device").agg(count("*") as "app_day_h_dev"), Seq("app", "day", "hour", "device"))
    trainDF.show

    //---------------------------------------------------------------------------

    //  ip|app|device| os|channel|label|hour|day|
    val categoricalColumns = Array("ip", "app", "device", "os", "channel", "hour","day")

    val numericCols = (trainDF.columns.toSet -- categoricalColumns.toSet - "label").toArray
//    val numericCols = (df_lessfeatures.columns.toSet -- categoricalColumns.toSet - "label").toArray

    var stages = new ArrayBuffer[org.apache.spark.ml.PipelineStage]()

    for (categoricalCol <- categoricalColumns) {
      
      val stringIndexer = new StringIndexer()
        .setInputCol(categoricalCol)
        .setOutputCol(categoricalCol+"_Index")
        .setHandleInvalid("keep")  //   options are "keep", "error" or "skip"
      
      //   Category Indexing with StringIndexer
      val encoder = new OneHotEncoder()
        .setInputCol(categoricalCol+"_Index")
        .setOutputCol(categoricalCol+"_Vec")

      stages = stages ++ Array(stringIndexer,encoder)
    }

    val assemblerInputs = categoricalColumns.map( {x:String => x + "_Vec"})  ++ numericCols
    val assembler = new VectorAssembler()
      .setInputCols(assemblerInputs)
      .setOutputCol("features")
    stages += assembler

    val preprocess_pipeline = new Pipeline()
      .setStages(stages.toArray)

    val preprocess_model: PipelineModel = preprocess_pipeline.fit(trainDF)

    // -- until this point we don't need to rebuild model again.

    // Get Feature Vector from Training Data
    val df_featuresvector = preprocess_model.transform(trainDF)
    df_featuresvector.show

    val labelfeatures = df_featuresvector.select("label", "features")
    labelfeatures.show
    
    // Save pipeline and pipeline model
    preprocess_model.write
      .overwrite()
      .save(output + "_pipeline_model")

    preprocess_pipeline.write
      .overwrite()
      .save(output + "_pipeline")

    labelfeatures.write
      .option("header","true")
      .mode(SaveMode.Overwrite)
      .format("parquet")
      .save(output + "_labelfeatures.parquet")

    sc.stop()

  }
}




