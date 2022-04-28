%pip install pyspark==3.1.1
from pyspark.ml.regression import RandomForestRegressor
from pyspark.sql import SparkSession
from pyspark.sql import SQLContext
from pyspark.ml.feature import StringIndexer
from pyspark.ml.linalg import Vectors
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.feature import VectorIndexer
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import VectorAssembler
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
#Creating Spark application and loading of dataset
spark = SparkSession.builder.appName('Linear Regression').getOrCreate()
load_data = spark.read.csv('/content/sample_data/weight-height.csv',inferSchema=True, header=True)
#To show the loaded dataframe
load_data.show()
#Converting into VectorFeature
get_assembler = VectorAssembler(inputCols=['Weight'],outputCol='features')
assembled_data = get_assembler.transform(load_data)
assembled_data.show()
finalized_data = assembled_data.selectExpr("features", "Height as label")
finalized_data = finalized_data.select("features", "label")
#To show the finalized dataframe
finalized_data.show()
#Splitting into training and testing dataset
training_data,testing_data = finalized_data.randomSplit([0.7,0.3])
rfr = RandomForestRegressor(featuresCol="features", maxDepth=30,numTrees=200)
# train the model
rfr_model = rfr.fit(training_data)
# select example rows to display.
predictions = rfr_model.transform(testing_data)
predictions.show(100)
# compute accuracy on the test set
evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(predictions)
print("Root Mean Squared Error (RMSE) = %g" % rmse)
