pivotDF = dataframe.groupBy().pivot("Department").sum("Wage")
pivotDF.show()

