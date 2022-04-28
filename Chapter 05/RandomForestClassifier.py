%pip install pyspark==3.1.1
from pyspark.ml.classification import RandomForestClassifier
from pyspark.sql import SparkSession
from pyspark.sql import SQLContext
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StringIndexer
from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml.feature import VectorIndexer
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
#Spark's application and loading of dataset
spark = SparkSession.builder.appName('trees').getOrCreate()
data = spark.read.csv(‘/content/sample_data/cancerdata.csv’,inferSchema=True, header=True)
#Dropping the rows that contains “Null Value”
data = data.dropna()
#Show the columns of dataframe
data.columns
#Converting the datatype into double or int
converteddata = data.selectExpr("cast(age as int) age",
    "cast(education as int) education",
    "cast(currentSmoker as double) currentSmoker",
    "cast(TenYearCHD as double) TenYearCHD",
	"cast(male as int) gender",
	"cast(cigsPerDay as double) cigsPerDay",
	"cast(BPMeds as double) BPMeds",
	"cast(prevalentStroke as double) prevalentStroke",
	"cast(prevalentHyp as double) prevalentHyp",
	"cast(diabetes as double) diabetes",
	"cast(totChol as double) totChol",
	"cast(sysBP as double) sysBP",
	"cast(diaBP as double) diaBP",
	"cast(BMI as double) BMI",
	"cast(heartRate as double) heartRate",
	"cast(glucose as double) glucose")
#Key features selection and converting into vectors
converteddata.dropna()
assembler = VectorAssembler(inputCols=['age', 'education', 'currentSmoker', 'gender','glucose','diabetes'],outputCol='get_feature')
get_output = assembler.transform(converteddata)
get_output.printSchema()
finalized_data = get_output.select("get_feature", "TenYearCHD")
training_data, testing_data = finalized_data.randomSplit([0.7,0.3])
rfc = RandomForestClassifier(labelCol="TenYearCHD", featuresCol="get_feature", numTrees=100, seed=50)
rfc_model = rfc.fit(training_data)
rfc_preds = rfc_model.transform(testing_data)
rfc_preds.show(5)
#evaluation Matrix
evaluator = MulticlassClassificationEvaluator(labelCol="TenYearCHD", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(rfc_preds)
print("Accuracy:", accuracy)
#Get summary of model and tree structure
treeModel = rfc_model
print(treeModel)
print(rfc_model.toDebugString)
