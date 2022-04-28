from pyspark.ml.feature import DCT
from pyspark.ml.linalg import Vectors
create_df = spark.createDataFrame([
    (Vectors.dense([5.0, 7.0]),),
    (Vectors.dense([3.0, 1.0]),)
], ["indispensable_features"])
get_dctfunc = DCT(inverse=False, inputCol="indispensable_features", outputCol="get_features")
dctDataFrame = get_dctfunc.transform(create_df)
dctDataFrame.select("get_features").show(truncate=False)


