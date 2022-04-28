dataframe.show()
dataframe.printSchema()
changed_dataframe = dataframe.withColumn("Wage",col("Wage").cast("string"))
changed_dataframe.printSchema()
