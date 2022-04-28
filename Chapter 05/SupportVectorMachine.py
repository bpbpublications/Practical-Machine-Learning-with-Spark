%pip install pyspark==3.1.1
from pyspark.ml.classification import LinearSVC
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql import SparkSession
from pyspark.sql import SQLContext
from pyspark.ml.feature import StringIndexer
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
#Creating Spark application and loading of dataset
spark = SparkSession.builder.appName('LSVM Regression').getOrCreate()
load_data = spark.read.csv('/content/sample_data/weight-height.csv',inferSchema=True, header=True)
#To show the loaded dataframe
load_data.show()
indexer = StringIndexer(inputCol='Gender', outputCol='label')
load_data = indexer.fit(load_data).transform(load_data)
#Converting into VectorFeature
get_assembler = VectorAssembler(inputCols=['Height', 'Weight'],outputCol='features')
assembled_data = get_assembler.transform(load_data)
assembled_data.show(5)
finalized_data = assembled_data.select("features", "label")
#To show the finalized dataframe
finalized_data.show(5)
#Splitting into training and testing dataset
training_data,testing_data = finalized_data.randomSplit([0.7,0.3])
# create the trainer and set its parameters
svmc = LinearSVC(maxIter=200, regParam=0.7)
# train the model
svmc_model = svmc.fit(training_data)
# select example rows to display.
predictions = svmc_model.transform(testing_data)
predictions.show(5)
# compute accuracy on the test set
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Accuracy  = " + str(accuracy))
