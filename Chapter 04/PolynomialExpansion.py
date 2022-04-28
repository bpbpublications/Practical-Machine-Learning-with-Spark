from pyspark.ml.feature import PolynomialExpansion
from pyspark.ml.linalg import Vectors
create_df = spark.createDataFrame([
    (Vectors.dense([5.0, 7.0]),),
    (Vectors.dense([3.0, 1.0]),)
], ["indispensable_features"])
polyfunc = PolynomialExpansion(degree=2, inputCol="indispensable_features", outputCol="get_Features")
polyfuncDF = polyfunc.transform(create_df)
polyfuncDF.show(truncate=False)

