Dataframe1 = sqlContext.read.format('com.databricks.spark.csv') \
.options(header='true', inferschema='true') \
.load('/home /Gourav/chap3/wage_table3.csv')
Dataframe1.show()
Distinct_DF = Dataframe1.distinct()
Distinct_DF.show()
