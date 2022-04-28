from pyspark.sql.functions import monotonically_increasing_id
dataframe = sqlContext.read.format('com.databricks.spark.csv') \
.options(header='true', inferschema='true') \
.load('/home /Gourav/chap3/wage_table3.csv')
get_dataframe =dataframe.withColumn("index",monotonically_increasing_id())
