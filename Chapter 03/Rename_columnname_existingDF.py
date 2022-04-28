renamed_df = dataframe.withColumnRenamed("gender","sex").show(truncate=False) 
renamed_df.printSchema()
