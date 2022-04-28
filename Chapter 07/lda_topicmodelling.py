from pyspark.ml import Pipeline
from pyspark.ml.feature import StopWordsRemover, CountVectorizer, IDF
from pyspark.ml.clustering import LDA
from pyspark.sql.functions import col, lit, concat, regexp_replace
from pyspark.sql.utils import AnalysisException
from pyspark.ml.feature import Tokenizer, RegexTokenizer
from pyspark.sql.functions import col, udf
from pyspark.sql.types import IntegerType
from pyspark.ml.clustering import LDA
from pyspark.ml.feature import StopWordsRemover
from pyspark.ml.feature import Normalizer
from pyspark.ml.linalg import Vectors

dataset = spark.read.format('com.databricks.spark.csv') \
.options(header='true', inferschema='true') \
.load('/home/cdh@psnet.com/Gourav/chap3/abcnews-date-text.csv')

get_tokenizers = Tokenizer(inputCol="headline_text", outputCol="get_tokens")
get_tokenized = get_tokenizers.transform(dataset)
remover = StopWordsRemover(inputCol="get_tokens", outputCol="row")
get_remover = remover.transform(get_tokenized)
counter_vectorized = CountVectorizer(inputCol="row", outputCol="get_features")
getmodel = counter_vectorized.fit(get_remover)
get_result = getmodel.transform(get_remover)
idf_function = IDF(inputCol="get_features", outputCol="get_idf_features")
train_model = idf_function.fit(get_result)
outcome = train_model.transform(get_result)
training,test = outcome.randomSplit([0.7,0.3])

lda = LDA(featuresCol='get_idf_features', k=10, maxIter=10)
model = lda.fit(training)
transformed = model.transform(test)
transformed.show(truncate=False)
# Describe topics.
topics = model.describeTopics(10)
print("The topics described by their top-weighted terms:")
topics.show(truncate=False)
# Shows the result
transformed = model.transform(test)
transformed.show(truncate=False)
