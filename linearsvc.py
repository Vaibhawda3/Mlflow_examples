#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
from __future__ import print_function
import os
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql.functions import udf
from pyspark.ml.evaluation import BinaryClassificationEvaluator
import mlflow
from mlflow import spark
import mlflow.pyfunc
from pyspark.sql.types import ArrayType,DoubleType
from pyspark.sql.types import IntegerType
# $example on$
from pyspark.ml.classification import LinearSVC
from urllib.parse import urlparse
from pyspark.ml.feature import VectorAssembler
# $example off$
from pyspark.ml.classification import RandomForestClassifier
from pyspark.sql import SparkSession
import logging
logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)
class UdfModelWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, ordered_df_columns, model_artifact):
        self.ordered_df_columns = ordered_df_columns
        self.model_artifact = model_artifact

    def load_context(self, context):
        import mlflow.pyfunc
        self.spark_pyfunc = mlflow.pyfunc.load_model(context.artifacts[self.model_artifact])

    def predict(self, context, model_input):
        renamed_input = model_input.rename(
            columns={
                str(index): column_name for index, column_name
                    in list(enumerate(self.ordered_df_columns))
            }
        )
        return self.spark_pyfunc.predict(renamed_input)

def log_udf_model(artifact_path, ordered_columns, run_id):
    udf_artifact_path = f"udf-{artifact_path}"
    data_path=f"udf-{artifact_path}"
    model_uri = f"runs:/{run_id}/{artifact_path}"
    mlflow.pyfunc.log_model(
        artifact_path = udf_artifact_path,
        code_path=[os.path.abspath(os.path.dirname(__file__))],
        python_model = UdfModelWrapper(ordered_columns, artifact_path),
        artifacts={ artifact_path: model_uri }
    )
    return udf_artifact_path

def train(run_id):
    spark = SparkSession.builder.appName("linearSVC Example").getOrCreate()
    training=spark.read.format("csv").load("/Dataset/titanic.csv")
    columns_to_drop=["_c2","_c7","_c8","_c9","_c10"]
    training=training.drop(*columns_to_drop)
    training=training.withColumn("_c0",training["_c0"].cast(IntegerType()))
    training=training.withColumn("_c1",training["_c1"].cast(IntegerType()))
    training=training.withColumn("_c3",training["_c3"].cast(IntegerType()))
    training=training.withColumn("_c4",training["_c4"].cast(IntegerType()))
    training=training.withColumn("_c5",training["_c5"].cast(IntegerType()))
    training=training.withColumn("_c6",training["_c6"].cast(IntegerType()))
    va = VectorAssembler(inputCols=["_c0", "_c1", "_c3","_c4","_c5","_c6"],outputCol="features",handleInvalid = "keep")
    training=va.transform(training)
    training=training.withColumnRenamed("_c0","label")
    #training=training.select([to_null(c).alias(c) for c in training.columns]).na.drop()
    training=training.dropna()
    train,test=training.randomSplit([0.7,0.3])
    maxiter=10
    regparam=0.1
    new_schema = ArrayType(DoubleType(), containsNull=False)
    udf_foo = udf(lambda x:x, new_schema)
    training = training.withColumn("features",udf_foo("features"))
    #lsvc = LinearSVC(maxIter=maxiter, regParam=regparam)
    rf = RandomForestClassifier(labelCol="label", featuresCol="features", numTrees=10)
    #mlflow.log_param("maxiter",maxiter)
    #mlflow.log_param("regparam",regparam)
    tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
    # Fit the modeltest.toPandas()
    #print(test["features"])
    #test["features"]=[eval(i) for i in list(test["features"])] 
    list_to_vector_udf = udf(lambda l: Vectors.dense(l), VectorUDT())
    df_with_vectors = training.select(
    training["label"],
    list_to_vector_udf(training["features"]).alias("features"))
    df_with_vectors.toPandas().to_csv("Test6.csv")
    lsvcModel = rf.fit(train)
    prediction=lsvcModel.transform(test)
    evaluator = BinaryClassificationEvaluator()
    roc=evaluator.evaluate(prediction, {evaluator.metricName: "areaUnderROC"})     # Print the coefficients and intercept for linear SVC
   # print("Coefficients: " + str(lsvcModel.coefficients))
   # print("Intercept: " + str(lsvcModel.intercept))
    mlflow.spark.save_model(lsvcModel,"my_new_rf_model1")
    mlflow.spark.log_model(lsvcModel,"rfModel1")
    mlflow.log_metric("roc",roc) 
    log_udf_model("rfModel1", training.columns, run_id)

if __name__ == "__main__":
        with mlflow.start_run() as run:
            print("MLflow:")
            print("  run_id:",run.info.run_id)
            train(run.info.run_id)

    # $example on$
    # Load training data
    #training = spark.read.format("libsvm").load("/Dataset/sample_libsvm_data.txt")
    #training=spark.read.csv("/Dataset/titanic.csv")
    # $example off$
    #spark.stop()
