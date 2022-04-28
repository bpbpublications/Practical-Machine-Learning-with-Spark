dataframe1 = sqlContext.read.format('com.databricks.spark.csv') \
.options(header='true', inferschema='true') \
.load('/home /Gourav/chap3/wage_table.csv')
dataframe2 = sqlContext.read.format('com.databricks.spark.csv') \
.options(header='true', inferschema='true') \
.load('/home /Gourav/chap3/wage_table2.csv')
Join_DF = dataframe1.join(dataframe2, on=['Department'], how='outer').show()
