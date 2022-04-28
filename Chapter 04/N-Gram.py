from pyspark.ml.feature import NGram
generate_df = spark.createDataFrame([
    (0, ["This" ,"Book", "Is", "For", "All", "The", "Big" , "Data" ,"And" ,"Data" ,"Science", "Lovers"]),
    (1, ["This" ,"Is" ,"Our", "Chapter-4" ,"Which" ,"Has", "Content", "Related", "To", "Spark", "MLlib"])], ["id", "create_df"])
get_ngram = NGram(n=2, inputCol="create_df", outputCol="get_ngram_out")
get_ngram_DataFrame = get_ngram.transform(generate_df)
get_ngram_DataFrame.select("get_ngram_out").show(truncate=False)
