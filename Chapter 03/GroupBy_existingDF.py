aggregated_df = dataframe.groupBy("Department").sum("Age").show(truncate=False)

##Aggregate functions with filter and group By 

dataframe.groupBy().sum("Wage").filter(F.col("Wage") >= 35).show(truncate=False)
dataframe.groupBy("Department").sum("Wage").show(truncate=False)
