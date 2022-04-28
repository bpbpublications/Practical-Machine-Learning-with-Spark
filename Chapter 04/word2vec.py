from pyspark.ml.feature import Word2Vec
Gen_DF = spark.createDataFrame([
    (0, "DataScience,MachineLearning,ApacheSpark,MachineLearning".split(",")),
    (1, "ApacheMLlib,MachineLearning,DataScience".split(","))], ["id", "words"])
func_word2Vec = Word2Vec(vectorSize=3, minCount=0, inputCol="words", outputCol="get_result")
model = func_word2Vec.fit(Gen_DF)
get_result = model.transform(Gen_DF)
get_result.show(truncate=False)

