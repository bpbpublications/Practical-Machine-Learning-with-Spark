%pip install pyspark==3.1.1
from pyspark.ml.classification import RandomForestClassifier
from spark_tree_plotting import plot_tree 
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
from spark_tree_plotting import plot_tree
from spark_tree_plotting import export_graphviz
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
#Spark’s application and loading of dataset
spark = SparkSession.builder.appName('trees').getOrCreate()
data = spark.read.csv('/content/sample_data/cancerdata.csv',inferSchema=True, header=True)
#Dropping the rows that contains "Null Value"
data = data.dropna()
#Show the columns of dataframe
data.columns
#Converting the datatype into double or int
converteddata = data.selectExpr(“cast(age as int) age”,
    “cast(education as int) education”,
    “cast(currentSmoker as double) currentSmoker”,
    “cast(TenYearCHD as double) TenYearCHD”,
	“cast(male as int) gender”,
	“cast(cigsPerDay as double) cigsPerDay”,
	“cast(BPMeds as double) BPMeds”,
	“cast(prevalentStroke as double) prevalentStroke”,
	“cast(prevalentHyp as double) prevalentHyp”,
	“cast(diabetes as double) diabetes”,
	“cast(totChol as double) totChol”,
	“cast(sysBP as double) sysBP”,
	“cast(diaBP as double) diaBP”,
	“cast(BMI as double) BMI”,
	“cast( eartrate as double)  eartrate”,
	“cast(glucose as double) glucose”)
#Key features selection and converting into vectors
converteddata.dropna()
assembler = VectorAssembler(inputCols=['age', 'education', 'currentSmoker', 'gender','glucose','diabetes'],outputCol='get_feature')
get_output = assembler.transform(converteddata)
get_output.printSchema()
finalized_data = get_output.select("get_feature", "TenYearCHD")
training_data, testing_data = finalized_data.randomSplit([0.7,0.3])
dtc = DecisionTreeClassifier(labelCol="TenYearCHD", featuresCol="get_feature")
dtc_model = dtc.fit(training_data)
dtc_preds = dtc_model.transform(testing_data)
dtc_preds.show()
#evaluation Matrix
evaluator = MulticlassClassificationEvaluator(labelCol="TenYearCHD", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(dtc_preds)
print("Accuracy:", accuracy)
#Get summary of model and tree structure
treeModel = dtc_model
print(treeModel)
print(dtc_model.toDebugString)
# Visualising the graph 
dec_tree = plot_tree(dtc_model, featureNames = ['age', 'education', 'currentSmoker', 'gender','glucose','diabetes'], classNames=[0,1], filled = True)
image = Image.open(io.BytesIO(dec_tree))
path_for_image = "/content/output"
image_name = path_for_image + "_" + ".png"
image.save(image_name)
