from pyspark.ml.feature import ChiSqSelector
from pyspark.ml.linalg import Vectors
create_df = spark.createDataFrame([
    (0, Vectors.dense([3.0, 6.0, 4.0]),5.0),
    (1, Vectors.dense([3.0, 8.1, 10.0]),7.0,)
], ["unique_id", "get_features", "label"])
selector = ChiSqSelector(numTopFeatures=2, featuresCol="get_features",
                         outputCol="selectedFeatures", labelCol="label")
get_result = selector.fit(create_df).transform(create_df)
get_result.show()