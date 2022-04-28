from pyspark.sql.functions import col
Dataframe = sqlContext.read.format('com.databricks.spark.csv') \
.options(header='true', inferschema='true') \
.load('/home /Gourav/chap3/wage_table2.csv')
selected_df=Dataframe.select("Department").sort("Wage").show()
get_sorted = Dataframe.sort(col("Age")).show(truncate=False)
