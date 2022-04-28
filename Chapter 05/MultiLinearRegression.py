%pip install pyspark==3.1.1
from pyspark.sql import SparkSession
from pyspark.sql import SQLContext
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StringIndexer
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import VectorIndexer
from pyspark.ml.linalg import Vectors
from pyspark.ml.regression import LinearRegression
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
#Loading and creation of Spark's application
spark = SparkSession.builder.appName('MultiLinearRegression').getOrCreate()
loaded_data = spark.read.csv('/content/sample_data/weatherHistory.csv',inferSchema=True, header=True)
#To check the columns of dataframe
loaded_data.columns
#Converting into the single Vector
get_assembler = VectorAssembler(inputCols=['Temperature (C)',
 'Apparent Temperature (C)',
 'Humidity',
 'Wind Speed (km/h)',
 'Wind Bearing (degrees)',
 'Visibility (km)',
 'Loud Cover',
 'Pressure (millibars)'],outputCol='get_feature')
op_assembler = get_assembler.transform(loaded_data)
op_assembler.show()
get_indexer = StringIndexer(inputCol='Summary', outputCol='summary_index')
finalized_data = get_indexer.fit(op_assembler).transform(op_assembler)
finalized_data = finalized_data.select("get_feature", "summary_index")
training_data, testing_data = finalized_data.randomSplit([0.7,0.3])
#Linear Regression function on multi-variant dataset
lr = LinearRegression(featuresCol="get_feature", labelCol="summary_index",)
lr_model = lr.fit(training_data)
#Evaluating the model on testing dataset to check the residue of each point
test_results = lr_model.evaluate(testing_data)
test_results.residuals.show()
#Testing dataset on lr_model
get_prediction = lr_model.transform(testing_data)
get_prediction.show()
#Get_training_insights
training_data.describe().show()
train = training_data.select("get_feature","summary_index").toPandas()
train_get_feature = train['get_feature']
train_get_feature = list(train_get_feature)
train_get_salary = train['summary_index']
#Training_Prediciton Insights
get_training_prediction = lr_model.transform(training_data)
#converting into Spark's df to Pandas's df for data visualization
train_pred = get_training_prediction.select("prediction").toPandas()
prediction_train = train_pred['prediction']
prediction_list = list(prediction_train)
print(prediction_list)
#Testing Insights
x = testing_data.select("get_feature","summary_index").toPandas()
x_get = x['get_feature']
y_get = x['summary_index']
#Get summary of the model
print("Summary of model is here:")
lr_model.summary
#Getting coefficients and intercept
print("Coefficients: " + str(lr_model.coefficients))
print("Intercept: " + str(lr_model.intercept))
#Evalutaion Metrics
eval = RegressionEvaluator(labelCol="summary_index", predictionCol="prediction", metricName="rmse")
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





