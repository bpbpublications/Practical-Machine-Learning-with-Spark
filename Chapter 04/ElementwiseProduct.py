from pyspark.ml.feature import ElementwiseProduct
from pyspark.ml.linalg import Vectors
create_df = spark.createDataFrame([
    (Vectors.dense([5.0, 7.0, 9.0]),),
    (Vectors.dense([3.0, 1.0, 6.0]),)
], ["indispensable_features"])
get_transformer = ElementwiseProduct(scalingVec=Vectors.dense([0.0, 1.0, 2.0]),inputCol="indispensable_features", outputCol="NewVector")
get_transformer.transform(create_df).show()
