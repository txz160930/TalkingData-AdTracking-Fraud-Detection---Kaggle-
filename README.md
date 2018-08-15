# bigdata_final_project


There are three executions phases:

./spark-submit --class "Preprocess" ml_2.11-0.1.jar  train_sample.csv ./out

./spark-submit --jars xgboost4j-spark-0.72-jar-with-dependencies.jar,xgboost4j-0.72-jar-with-dependencies.jar --class "XGBoostML" ml_2.11-0.1.jar  ./out_labelfeatures.parquet/ ./out_pipeline_model/ out_test/ result

./spark-submit --class "Evaluation" ml_2.11-0.1.jar  result_predicts.parquet/

// Updated Commands

==================================================================================

Preprocessing Data

./spark-submit --class "Preprocess" ml_2.11-0.1.jar  train_sample.csv ./out


==================================================================================

./spark-submit --jars xgboost4j-spark-0.72-jar-with-dependencies.jar,xgboost4j-0.72-jar-with-dependencies.jar --class "XGBoostML" ml_2.11-0.1.jar  ./out_labelfeatures.parquet/ ./out_pipeline_model/ out_test/ result_xgb

./spark-submit --class "RandomForest" ml_2.11-0.1.jar ./out_labelfeatures.parquet/ ./out_pipeline_model/ out_test/ result_rf

./spark-submit --class "GBT" ml_2.11-0.1.jar  ./out_labelfeatures.parquet/ ./out_pipeline_model/ out_test/ result_gbt


==================================================================================

Evaluation

./spark-submit --class "Evaluation" ml_2.11-0.1.jar  result_xgb_predicts.parquet/ eval_res_xgb

./spark-submit --class "Evaluation" ml_2.11-0.1.jar  result_rf_predicts.parquet/ eval_res_rf

./spark-submit --class "Evaluation" ml_2.11-0.1.jar  result_gbt_predicts.parquet/ eval_res_gbt




To compile xgboost (latest version 0.72) and publish to local maven directory, please follow these steps:


source: 

https://docs.databricks.com/user-guide/faq/xgboost.html

https://xgboost.readthedocs.io/en/latest/jvm/index.html

sudo apt-get update

sudo apt-get install -y maven

sudo apt-get install -y cmake

// You can skip this step if you have already install java 8

export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64/

git clone --recursive https://github.com/dmlc/xgboost

cd xgboost/jvm-packages

mvn -DskipTests=true -Dspark.version=2.2.1 package

mvn -DskipTests=true install



