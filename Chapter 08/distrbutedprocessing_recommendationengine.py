>>from pyspark.sql import SparkSession
>>spark = SparkSession.builder.appName("recommendation").getOrCreate()
>>from pyspark.ml.recommendation import ALS
>>from pyspark.ml.evaluation import RegressionEvaluator
>>from pyspark.ml.feature import StringIndexer
>>dataset = spark.read.format('com.databricks.spark.csv') \
.options(header='true', inferschema='true') \
.load('/home/cdh@psnet.com/Gourav/ml-25m/ratings.csv')
>>dataset_refined = dataset.withColumn('rating', dataset.rating.cast('integer')).dropna()
>>training, testing = dataset_refined.randomSplit([0.7,0.3])
>>als = ALS(maxIter=10,regParam=0.05,userCol='userId',itemCol='movieId', ratingCol='rating')
>>model = als.fit(training)
>>predictions = model.transform(training).show()
>>predictions = model.transform(training)
#validation of the model
>>test_prediction = model.transform(testing).show()
>>test_prediction = model.transform(testing)
#saving the result
>>test_prediction.select('userId','movieId', 'rating', 'prediction').write.csv('/home/cdh@psnet.com/Gourav/recommendation/')
>>evaluator = RegressionEvaluator(metricName='rmse', labelCol='rating', predictionCol='prediction')
>>rmse = evaluator.evaluate(test_prediction)
