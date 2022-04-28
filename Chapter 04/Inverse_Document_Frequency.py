from pyspark.ml.feature import HashingTF
from pyspark.ml.feature import HashingTF, IDF, Tokenizer
Gen_DF = spark.createDataFrame([
    (0, "DataScience,MachineLearning,ApacheSpark,MachineLearning".split(",")),
    (1, "ApacheMLlib,MachineLearning,DataScience".split(","))
], ["id", "words"])
gen_HF = HashingTF(inputCol="words", outputCol="features", numFeatures=100)
get_HTF = gen_HF.transform(Gen_DF)
idf_function = IDF(inputCol="features", outputCol="get_idf_feature")
train_model = idf_function.fit(get_HTF)
outcome = train_model.transform(get_HTF)
outcome.show(truncate=False)

