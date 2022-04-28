from pyspark.ml.feature import QuantileDiscretizer
create_df = spark.createDataFrame([(1001, 180000.0), (1003, 190000.0), (1004, 800000.0), (3002, 500000.0), (4871, 7000000.0)], ["employee_id", "salary"])
quant_discretizer = QuantileDiscretizer(numBuckets=3, inputCol="salary", outputCol="result")
get_result = quant_discretizer.fit(create_df).transform(create_df)
get_result.show()


