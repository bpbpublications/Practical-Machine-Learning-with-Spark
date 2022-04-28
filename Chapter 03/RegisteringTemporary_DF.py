import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit
from pyspark.sql.types import StructType, StructField, StringType,IntegerType
dataframe = sqlContext.read.format('com.databricks.spark.csv') \
.options(header='true', inferschema='true') \
.load('/home /Gourav/chap3/wage_table3.csv')
dataframe.show(5)
dataframe.registerTempTable("get_table")
sqlContext.sql("select * from get_table").show(5)
