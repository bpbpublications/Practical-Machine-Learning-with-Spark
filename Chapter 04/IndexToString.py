from pyspark.ml.feature import StringIndexer
create_df = spark.createDataFrame(
    [(0, "Hello"), (1, "All"), (2, "This"), (3, "is"), (4, "a"), (5, "new"), (6, "Day")],
    ["unique_id", "words"])
stage1_output = StringIndexer(inputCol="words", outputCol="Conversion_outcome")
get_finalized = stage1_output.fit(create_df).transform(create_df)
get_finalized.show()

