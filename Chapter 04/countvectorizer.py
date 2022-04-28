from pyspark.ml.feature import CountVectorizer
Gen_DF = spark.createDataFrame([
    (0, "DataScience,MachineLearning,ApacheSpark,MachineLearning".split(",")),
    (1, "ApacheMLlib,MachineLearning,DataScience".split(","))
], ["id", "words"])
counter_vectorized = CountVectorizer(inputCol="words", outputCol="get_features")
getmodel = counter_vectorized.fit(Gen_DF)
get_result = getmodel.transform(Gen_DF)
get_result.show(truncate=False)
