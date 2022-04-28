from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
create_df = spark.createDataFrame([
    (0, Vectors.dense([3.0, 6.0, -4.0]), 18),
    (1, Vectors.dense([3.0, 8.1, 10.0]), 30)
], ["unique_id", "get_features", "user_age"])
get_assembler = VectorAssembler(inputCols=["get_features", "user_age"], outputCol="features")
result = get_assembler.transform(create_df)
result.show(truncate=False)

