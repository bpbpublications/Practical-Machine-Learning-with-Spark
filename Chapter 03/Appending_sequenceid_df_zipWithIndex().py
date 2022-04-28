schema_rdd = dataframe.rdd.zipWithIndex().map(lambda (row,rowId): ( list(row) + [rowId+1]))
indexed_df = sqlContext.createDataFrame(schema_rdd, schema=dataframe_schema.schema)
indexed_df.printSchema()
indexed_df.show(10)
or
indexed_df.registerTempTable("registered_table")
sqlContext.sql("select * from registered_table").show(10)
