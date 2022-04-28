upadted_dataframe = dataframe.withColumn("Wage",col("Wage")*2)
upadted_dataframe.show()
updated_dataframe.printSchema()

