from pyspark.ml.feature import Tokenizer, RegexTokenizer
from pyspark.sql.functions import col, udf
from pyspark.sql.types import IntegerType
generate_df = spark.createDataFrame([
    (0, "This Book Is For All The Big Data And Data Science Lovers"),
    (1, "This Is Our Chapter-4 Which Has Content Related To Spark MLlib ")], ["id", "create_df"])
get_tokenizers = Tokenizer(inputCol="create_df", outputCol="get_tokens")
get_tokenized = get_tokenizers.transform(generate_df)
remover = StopWordsRemover(inputCol="get_tokens", outputCol="row")
remover.transform(get_tokenized).select("get_tokens", "row").show(truncate=False)
