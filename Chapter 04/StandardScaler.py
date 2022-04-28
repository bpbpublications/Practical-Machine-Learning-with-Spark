from pyspark.ml.feature import StandardScaler
create_df = spark.createDataFrame([
    (0, Vectors.dense([3.0, 6.0, -4.0]), 18),
    (1, Vectors.dense([3.0, 8.1, 10.0]), 30),
    (2, Vectors.dense([0.0, 19.1, 16.0]), 60)
], ["unique_id", "get_features", "user_age"])
get_scaler = StandardScaler(inputCol="get_features", outputCol="scaled_ouput", withStd=True, withMean=False)
train_model = get_scaler.fit(create_df)
output_scaledD = train_model.transform(create_df)
output_scaledD.show(truncate=False)


