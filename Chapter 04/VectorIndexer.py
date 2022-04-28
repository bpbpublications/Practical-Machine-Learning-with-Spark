from pyspark.ml.feature import ChiSqSelector
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorIndexer, VectorAssembler
create_df = spark.createDataFrame([
    (0, Vectors.dense([3.0, 6.0, -4.0]), 18.0),
    (1, Vectors.dense([3.0, 2.0, 10.0]), 30.0)
], ["unique_id", "get_features", "user_age"])
vector_ind = VectorIndexer(inputCol="get_features", outputCol="get_result")
encode = vector_ind.fit(create_df).transform(create_df)
encode.show()

