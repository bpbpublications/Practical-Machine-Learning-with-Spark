from pyspark.sql import SQLContext
sqlContext = SQLContext(sc)
df = sqlContext.read.format('com.databricks.spark.csv').options(header='true', inferschema='true').load(‘/home /Gourav/chap3/us-500.csv’) # this is your csv file
df.show()
get_map_transform = df.select(df.columns[0]).rdd.map(lambda x: (x,1))
get_map_transform.take(10)
