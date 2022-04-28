from pyspark.ml.feature import HashingTF
Gen_DF = spark.createDataFrame([
    (0, "DataScience,MachineLearning,ApacheSpark,MachineLearning".split(",")),
    (1, "ApacheMLlib,MachineLearning,DataScience".split(","))
], ["id", "words"])
gen_HF = HashingTF(inputCol="words", outputCol="features")
get_result = gen_HF.transform(Gen_DF)
get_result.show(truncate=False)
