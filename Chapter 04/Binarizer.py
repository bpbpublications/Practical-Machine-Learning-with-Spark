from pyspark.ml.feature import Binarizer
from pyspark.ml.feature import StringIndexer
create_df = spark.createDataFrame(
    [(0, "Hello"), (1, "All"), (2, "This"), (3, "is"), (4, "a"), (5, "new"), (6, "Day")],
    ["unique_id", "words"])
stage1_output = StringIndexer(inputCol="words", outputCol="Conversion_outcome")
get_finalized_df = stage1_output.fit(create_df).transform(create_df)
binarizer_value = Binarizer(threshold=3, inputCol="Conversion_outcome", outputCol="get_binarized_feature")
binarizedDF = binarizer_value.transform(get_finalized_df)
binarizedDF.show()
