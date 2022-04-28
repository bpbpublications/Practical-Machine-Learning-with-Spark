dataframe = sqlContext.read.format('com.databricks.spark.csv') \
.options(header='true', inferschema='true') \
.load('/home /Gourav/chap3/wage_table3.csv')
dropMulDF = dataframe.dropDuplicates(["Department","Age"])
print("Distinct count of department & Name : "+str(dropMulDF.count()))
dropMulDF.show(truncate=False)
