from pyspark.ml.feature import Tokenizer, RegexTokenizer
from pyspark.sql.functions import col, udf
from pyspark.sql.types import IntegerType
generate_df = spark.createDataFrame([
    (0, "This Book Is For All The Big Data And Data Science Lovers"),
    (1, "This Is Our Chapter-4 Which Has Content Related To Spark MLlib ")], ["unique_id", "generate_df"])
get_tokenizers = Tokenizer(inputCol="generate_df", outputCol="get_tokens")
get_tokenized = get_tokenizers.transform(generate_df)
#Display Outcome
get_tokenized.select("generate_df", "get_tokens").show(truncate=False)
#Save into Parquet Format
get_tokenized.select("generate_df", "get_tokens").write.save("parquetfileformat")
#Save Outcome Into a JSON Format
get_tokenized.select("generate_df", "get_tokens").write.json("JsonSave.json")

