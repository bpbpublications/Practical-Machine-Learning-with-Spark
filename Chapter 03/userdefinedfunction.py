from pyspark.sql import SQLContext
from pyspark.sql.types import *
from pyspark.sql.functions import udf
from pyspark.sql import Row
def new_udf(x):
   new_row = x.lower()
   return new_row
dataframe1 = sqlContext.read.format('com.databricks.spark.csv') \
.options(header='true', inferschema='true') \
.load('/home /Gourav/chap3/wage_table.csv')
updated_udf = udf(new_udf, StringType())
updated_df = dataframe1.withColumn('Department', updated_udf(dataframe1['Department']))
