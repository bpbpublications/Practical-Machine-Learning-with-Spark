from pyspark.ml.feature import VectorSlicer
from pyspark.ml.linalg import Vectors
from pyspark.sql.types import Row
df = spark.createDataFrame([
    (0, Vectors.dense([3.0, 6.0, -4.0]),),
    (1, Vectors.dense([3.0, 8.1, 10.0]),)
], ["unique_id", "get_features"])
slicer = VectorSlicer(inputCol="get_features", outputCol="features", indices=[0,2,1])
output = slicer.transform(df)
output.show(truncate=False)”
