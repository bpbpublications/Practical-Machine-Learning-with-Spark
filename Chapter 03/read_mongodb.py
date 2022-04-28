Spark_Initiate = SparkSession.builder.appName(‘Mongo-Spark Bridge’) \
 .config(‘spark.mongodb.input.uri’, ‘mongodb://127.0.0.1/analytics.get_insights’)  \
.getOrCreate()
df = spark.read.format(‘com.mongodb.spark.sql.DefaultSource’).load()
df.createOrReplaceTempView(‘get_insights’)
getDF = spark.sql(‘select * from get_insights’)
get_DF.show()
