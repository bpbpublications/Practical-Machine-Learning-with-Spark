%pip install pyspark==3.1.1
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
spark = SparkSession.builder.appName('Ridge Linear Regression').getOrCreate()
load_data = spark.read.csv('/content/sample_data/weight-height.csv',inferSchema=True, header=True)
#To show the loaded dataframe
load_data.show()
#Converting into VectorFeature
get_assembler = VectorAssembler(inputCols=['Height'],outputCol='feature')
assembled_data = get_assembler.transform(load_data)
assembled_data.show()
finalized_data = assembled_data.select("feature", "Weight")
#To show the finalized dataframe
finalized_data.show()
#Splitting into training and testing dataset
training_data,testing_data = finalized_data.randomSplit([0.7,0.3])
#Calling of LinearRegression function
lr = LinearRegression(featuresCol="feature", labelCol="Weight", elasticNetParam=1.0,regParam=0.5,maxIter=50,solver='normal'  )
lr_model = lr.fit(training_data)
#Evaluating the model on testing dataset to check the residue of each point
test_results = lr_model.evaluate(testing_data)
test_results.residuals.show()
#Testing dataset on lr_model
get_prediction = lr_model.transform(testing_data)
get_prediction.show()
#Get_training_insights
training_data.describe().show()
train = training_data.select("feature","Weight").toPandas()
train_get_feature = train['feature']
train_get_feature = list(train_get_feature)
train_get_salary = train['Weight']
#Training_Prediciton Insights
get_training_prediction = lr_model.transform(training_data)
#converting into Spark's df to Pandas's df for data visualization
train_pred = get_training_prediction.select("prediction").toPandas()
prediction_train = train_pred['prediction']
prediction_list = list(prediction_train)
print(prediction_list)
#Testing Insights
x = testing_data.select("feature","Weight").toPandas()
x_get = x['feature']
y_get = x['Weight']
#Get summary of the model
print("Summary of model is here:")
lr_model.summary
#Getting coefficients and intercept
print("Coefficients: " + str(lr_model.coefficients))
print("Intercept: " + str(lr_model.intercept))
#Visualization
plt.scatter(list(x_get), list(y_get), color = 'red')
plt.plot(train_get_feature, prediction_list, color = 'blue')
plt.title('Weight vs Height (Test set)')
plt.xlabel('Height')
plt.ylabel('Weight')
plt.show()
#Evalutaion Metrics
eval = RegressionEvaluator(labelCol="Weight", predictionCol="prediction", metricName="rmse")
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
