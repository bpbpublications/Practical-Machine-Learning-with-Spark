%pip install pyspark==3.1.1
from pyspark.ml.classification import LogisticRegression
from pyspark.sql import SparkSession
from pyspark.sql import SQLContext
from pyspark.ml.feature import StringIndexer
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
#Creating Spark application and loading of dataset
spark = SparkSession.builder.appName('Logistic Regression').getOrCreate()
load_data = spark.read.csv('/content/sample_data/weight-height.csv',inferSchema=True, header=True)
#To show the loaded dataframe
load_data.show(5)
#To convert stringintovector
indexer = StringIndexer(inputCol='Gender', outputCol='label')
final_data = indexer.fit(load_data).transform(load_data)
#Converting into VectorFeature
get_assembler = VectorAssembler(inputCols=['Height', 'Weight'],outputCol='features')
assembled_data = get_assembler.transform(final_data)
assembled_data.show(5)
finalized_data = assembled_data.select("features", "label")
#To show the finalized dataframe
finalized_data.show(5)
#Splitting into training and testing dataset
training_data,testing_data = finalized_data.randomSplit([0.7,0.3])
# Load training data
lr = LogisticRegression(maxIter=150, regParam=0.3, elasticNetParam=0.4)
# Fit the model
lrModel = lr.fit(training_data)
get_result = lrModel.transform(testing_data)
get_result.show(5)
# Print the coefficients and intercept for multinomial logistic regression
print("Coefficients: \n" + str(lrModel.coefficientMatrix))
print("Intercept: " + str(lrModel.interceptVector))
trainingSummary = lrModel.summary
# Obtain the objective per iteration
ObjHist = trainingSummary.objectiveHistory
print("ObjHist:")
for obj in ObjHist:
   print(obj)
# for multiclass, we can inspect metrics on a per-label basis
print("False positive rate by label:")
for i, rate in enumerate(trainingSummary.falsePositiveRateByLabel):
    print("label %d: %s" % (i, rate))
print("True positive rate by label:")
for i, rate in enumerate(trainingSummary.truePositiveRateByLabel):
   print("label %d: %s" % (i, rate))
print("Precision by label:")
for i, prec in enumerate(trainingSummary.precisionByLabel):
    print("label %d: %s" % (i, prec))
print("Recall by label:")
for i, rec in enumerate(trainingSummary.recallByLabel):
    print("label %d: %s" % (i, rec))
print("F-measure by label:")
for i, f in enumerate(trainingSummary.fMeasureByLabel()):
    print("label %d: %s" % (i, f))
accuracy = trainingSummary.accuracy
falsePositiveRate = trainingSummary.weightedFalsePositiveRate
truePositiveRate = trainingSummary.weightedTruePositiveRate
fMeasure = trainingSummary.weightedFMeasure()
precision = trainingSummary.weightedPrecision
recall = trainingSummary.weightedRecall
print("Accuracy: %s\nFPR: %s\nTPR: %s\nF-measure: %s\nPrecision: %s\nRecall: %s"
      % (accuracy, falsePositiveRate, truePositiveRate, fMeasure, precision, recall))
#Visualize ROC Curve
import matplotlib.pyplot as plt
plt.figure(figsize=(5,5))
plt.plot([0, 1], [0, 1], 'r--')
plt.plot(lrModel.summary.roc.select('FPR').collect(),
         lrModel.summary.roc.select('TPR').collect())
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.show()
