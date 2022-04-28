%pip install pyspark==3.1.1
from pyspark.ml.regression import IsotonicRegression
from pyspark.sql import SparkSession
from pyspark.sql import SQLContext
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import VectorIndexer
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
#Creating Spark application and loading of dataset
spark = SparkSession.builder.appName('Generalized Linear Regression').getOrCreate()
load_data = spark.read.csv('/content/sample_data/weight-height.csv',inferSchema=True, header=True)
#Converting into VectorFeature
get_assembler = VectorAssembler(inputCols=['Height'],outputCol='feature')
assembled_data = get_assembler.transform(load_data)
finalized_data = assembled_data.select("feature", "Weight")
finalized_data = finalized_data.selectExpr("feature as features", "Weight as label")
#Splitting into training and testing dataset
training_data,testing_data = finalized_data.randomSplit([0.7,0.3])
iso_reg = IsotonicRegression()
# Fit the model
iso_model = iso_reg.fit(training_data)
#Testing dataset on lr_model
get_prediction = iso_model.transform(testing_data)
get_prediction.show()
#Get_training_insights
train = training_data.select("features","label").toPandas()
train_get_feature = train['features']
train_get_feature = list(train_get_feature)
train_get_salary = train['label']
#Training_Prediciton Insights
get_training_prediction = iso_model.transform(training_data)
#converting into Spark's df to Pandas's df for data visualization
train_pred = get_training_prediction.select("prediction").toPandas()
prediction_train = train_pred['prediction']
prediction_list = list(prediction_train)
print(prediction_list)
#Testing Insights
x = testing_data.select("features","label").toPandas()
x_get = x['features']
y_get = x['label']
#Visualization
plt.scatter(list(x_get), list(y_get), color = 'red')
plt.plot(train_get_feature, prediction_list, color = 'blue')
plt.title('Weight vs Height (Test set)')
plt.xlabel('Height')
plt.ylabel('Weight')
plt.show()
#Evalutaion Metrics
eval = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse")
# Root Mean Square Error
rmse = eval.evaluate(get_prediction)
print("RMSE: %.3f" % rmse)
# Mean Square Error
mse = eval.evaluate(get_prediction, {eval.metricName: "mse"})
print("MSE: %.3f" % mse)
# Mean Absolute Error
mae = eval.evaluate(get_prediction, {eval.metricName: "mae"})
print("MAE: %.3f" % mae)
# r2 - coefficient of determination
r2 = eval.evaluate(get_prediction, {eval.metricName: "r2"})
print("r2: %.3f" %r2)
