new_column = dataframe.withColumn("New Column",col("Age")* 3)
new_column.printSchema()
new_column.show(10)
                             
#Adding a new column using constant value using lit function
integrated_litfunc = dataframe.withColumn("lit_column", lit("200"))
integrated.show(10)
