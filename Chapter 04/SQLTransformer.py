from pyspark.ml.feature import SQLTransformer
create_df = spark.createDataFrame([
    (0, Vectors.dense([3.0, 6.0, -4.0]), 18),
    (1, Vectors.dense([3.0, 8.1, 10.0]), 30),
    (2, Vectors.dense([0.0, 19.1, 16.0]), 60)
], ["unique_id", "get_features", "user_age"])
sqlTrfunc = SQLTransformer(statement="SELECT get_features from __THIS__ where user_age <= 30")
sqlTrfunc.transform(create_df).show()


