%pip install pyspark==3.1.1
from pyspark.ml.classification import LinearSVC
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql import SparkSession
from pyspark.sql import SQLContext
from pyspark.ml.feature import StringIndexer
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.linalg import Vectors
from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.feature import VectorIndexer
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import VectorAssembler
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
#Creating Spark application and loading of dataset
spark = SparkSession.builder.appName('Gradient Boosted Tree Regression').getOrCreate()
load_data = spark.read.csv('/content/sample_data/weight-height.csv',inferSchema=True, header=True)
#To show the loaded dataframe
load_data.show(5)
indexer = StringIndexer(inputCol='Gender', outputCol='label')
load_data = indexer.fit(load_data).transform(load_data)
#Converting into VectorFeature
get_assembler = VectorAssembler(inputCols=['Height', 'Weight'],outputCol='features')
assembled_data = get_assembler.transform(load_data)
assembled_data.show(5)
finalized_data = assembled_data.select(“features”, “label”)
#To show the finalized dataframe
finalized_data.show(5)
#Splitting into training and testing dataset
training_data,testing_data = finalized_data.randomSplit([0.7,0.3])
dtr = GBTRegressor(featuresCol="features", maxDepth=15, seed=40,labelCol="label",stepSize=0.7)
# train the model
dtr_model = dtr.fit(training_data)
# select example rows to display.
predictions = dtr_model.transform(testing_data)
predictions.show(5)
# compute accuracy on the test set
evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(predictions)
print("Root Mean Squared Error (RMSE) = %g" % rmse)
