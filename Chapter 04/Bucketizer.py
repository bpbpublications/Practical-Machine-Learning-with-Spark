from pyspark.ml.feature import Bucketizer
splits = [ -float("inf"), -0.5, 0.0, 0.5, 1.0, 2.0, float("inf")]
create_df = spark.createDataFrame([(-0.5,), (-0.3,), (0.0,), (1.0,),(0.2,), (100.0,)], ["get_features"])
apply_func_bucketizer = Bucketizer(splits=splits, inputCol="get_features", outputCol="buckfeatures")
get_data = apply_func_bucketizer.transform(create_df)
get_data.show()

