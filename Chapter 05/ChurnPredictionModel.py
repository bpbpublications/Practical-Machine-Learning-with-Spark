%pip install pyspark==3.1.1
from pyspark.sql import SparkSession
from pyspark.sql import SQLContext
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline
from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml.feature import VectorIndexer
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import matplotlib.pyplot as plt
import pandas as pd
#Creating Spark’s application and loading of dataset
spark = SparkSession.builder.appName('trees').getOrCreate()
get_data = spark.read.csv('/content/sample_data/Churn_Modelling.csv',inferSchema=True, header=True)
#To check the name of columns of dataframe
get_data.columns
#Converting the features into vector space features
get_assembler = VectorAssembler(inputCols=[
 'CreditScore',
 'Age',
 'Tenure',
 'Balance',
 'NumOfProducts',
 'HasCrCard',
 'IsActiveMember',
 'EstimatedSalary'],outputCol='get_feature')
assembled_data = get_assembler.transform(get_data)
assembled_data.show()
finalized_data = assembled_data.select("get_feature", "Exited")
training_data, testing_data = finalized_data.randomSplit([0.7,0.3])
#Calling DecisionClassifier class for training the model
dtc = DecisionTreeClassifier(labelCol="Exited", featuresCol="get_feature")
dtc_model = dtc.fit(training_data)
dtc_preds = dtc_model.transform(testing_data)
dtc_preds.show()
#Evaluation
evaluator = MulticlassClassificationEvaluator(labelCol="Exited", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(dtc_preds)
print(“Accuracy of the model on testing dataset:”,accuracy)
#Summary of the model
get_tree_summary = dtc_model
print(get_tree_summary)
