from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import (VectorSizeHint, VectorAssembler)
create_df = spark.createDataFrame([
    (0, Vectors.dense([3.0, 6.0, -4.0]), 18),
    (1, Vectors.dense([3.0, 10.0]), 30)
], ["unique_id", "get_features", "user_age"])
get_assembler = VectorAssembler(inputCols=["get_features", "user_age"], outputCol="features")
Vec_Si_Hi = VectorSizeHint(
    inputCol="get_features",
    handleInvalid="error",
    size=3)
get_dataset= Vec_Si_Hi.transform(create_df)
get_dataset.show(truncate=False)


