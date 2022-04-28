from pyspark.ml.feature import FeatureHasher
createDF = spark.createDataFrame([
    (10, "100", True, "Data Science"),
    (20, "200", False, "Big Data"),
    (30, "300", True, "Machine Learning with Spark"),
    (40, "400", False, "Deep Learning")
], ["col1", "col2", "col3", "col4"])
get_hasher = FeatureHasher(inputCols=["col1", "col2", "col3", "col4"],
                       outputCol="features", numFeatures = 10)
get_result = get_hasher.transform(createDF)
get_result.show(truncate=False)
