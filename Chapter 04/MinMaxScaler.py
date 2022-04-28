from pyspark.ml.feature import MinMaxScaler
from pyspark.ml.linalg import Vectors
create_df = spark.createDataFrame([
    (0, Vectors.dense([3.0, 6.0, -4.0]),),
    (1, Vectors.dense([3.0, 8.1, 10.0]),)
], ["unique_id", "get_features"])
get_scaler = MinMaxScaler(inputCol="get_features", outputCol="feature_outcome")
Train_Model = get_scaler.fit(create_df)
scaled_result = Train_Model.transform(create_df)
scaled_result.select("get_features", "feature_outcome").show()


