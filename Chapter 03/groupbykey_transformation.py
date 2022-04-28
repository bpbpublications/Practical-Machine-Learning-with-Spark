Dataframe1 = sqlContext.read.format('com.databricks.spark.csv') \
.options(header='true', inferschema='true') \
.load('/home /Gourav/chap3/wage_table3.csv')
Dataframe1.show()
DataFrame1.createOrReplaceTempView("new_df")
transformed_DF = spark.sql("select Department, sum(Wage) from new_df group by Department")
transformed_DF.show()
Or
Dataframe1.groupBy("Department").sum("Wage").show(false)
