%pip install pyspark==3.1.1
from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.classification import GBTClassifier
from pyspark.sql import SparkSession
from pyspark.sql import SQLContext
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import VectorIndexer
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
#Spark's application and loading of dataset
spark = SparkSession.builder.appName('Gradient Boosted Tree Classifier').getOrCreate()
data = spark.read.csv(‘/content/sample_data/cancerdata.csv’,inferSchema=True, header=True)
#Dropping the rows that contains “Null Value”
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
assembler = VectorAssembler(inputCols=['age', 'education', 'currentSmoker', 'gender','glucose','diabetes'],outputCol='features')
get_output = assembler.transform(converteddata)
get_output.printSchema()
finalized_data = get_output.select("features", "TenYearCHD")
finalized_data = finalized_data.selectExpr("features", "TenYearCHD as label")
training_data, testing_data = finalized_data.randomSplit([0.7,0.3])
#The input represents the number of features to be used for training a model
# Last element in layers represents the number of classes to be used for output
# Between first and last elements of layers must be for intermediate processing
trainer = MultilayerPerceptronClassifier(maxIter=150, layers=[6, 5 , 4, 6, 4, 2], blockSize=64, seed=50)
# training and testing of the model
model = trainer.fit(training_data)
result = model.transform(testing_data)
result.show(5)
#Evaluation
predictionAndLabels = result.select("prediction", "label")
evaluator = MulticlassClassificationEvaluator(metricName="accuracy")
print("Test set accuracy = " + str(evaluator.evaluate(predictionAndLabels)))
