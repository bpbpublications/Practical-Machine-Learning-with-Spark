from pyspark.ml.feature import Imputer
create_df = spark.createDataFrame([
    (0, Vectors.dense([3.0, 6.0, -4.0]), 18.0),
    (1, Vectors.dense([3.0, 8.1, 10.0]), 30.0),
    (2, Vectors.dense([0.0, 19.1, 16.0]), float("nan"))
], ["unique_id", "get_features", "user_age"])
get_imputer = Imputer(inputCols=["user_age"], outputCols=["Result_a"])
get_model = get_imputer.fit(create_df)
get_model.transform(create_df).show()

