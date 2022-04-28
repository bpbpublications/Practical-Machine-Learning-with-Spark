from pyspark.sql import SQLContext
sqlContext = SQLContext(sc)
df_csv = sqlContext.read.format('com.databricks.spark.csv') \
.options(header='true', inferschema='true') \
.load('/home /Gourav/chap3/us-500.csv') # this is path of  csv file
df_csv.show(5)

