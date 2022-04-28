from pyspark.ml.feature import Normalizer
from pyspark.ml.linalg import Vectors
create_df = spark.createDataFrame([
    (Vectors.dense([4.0, 2.0]),),
    (Vectors.dense([3.0, 1.0]),)
], ["indispensable_features"])
get_normalized = Normalizer(inputCol="indispensable_features", outputCol="Get_Features", p=1.0)
NormDataDF = get_normalized.transform(create_df)
NormDataDF.show()


