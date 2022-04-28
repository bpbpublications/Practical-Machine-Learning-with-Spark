Dataframe = sqlContext.read.format('com.databricks.spark.csv') \
.options(header='true', inferschema='true') \
.load('/home /Gourav/chap3/wage_table2.csv')
Dataframe.show()
Dataframe_sampled = Dataframe.sample(False, 0.7)
Dataframe_sampled.show()

