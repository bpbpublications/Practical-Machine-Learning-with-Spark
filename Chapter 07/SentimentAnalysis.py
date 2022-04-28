from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import StopWordsRemover, CountVectorizer, IDF, StringIndexer
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
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import NaiveBayes
from pyspark.ml import Pipeline
from pyspark.sql.functions import length
#Read data from a CSV
spark = SparkSession.builder.appName('nlp').getOrCreate()
dataset = spark.read.format('com.databricks.spark.csv') \
.options(header='true', inferschema='true') \
.load('/home/cdh@psnet.com/Gourav/chap3/Review.csv')
#DataRefining
dataset_refined = dataset.withColumn('Liked', dataset.Liked.cast('integer'))
dataset_refined =  dataset_refined.selectExpr("Review as review", "Liked as label")
data_length = dataset_refined.withColumn('length', length(dataset_refined['review']))
tokenizer = Tokenizer(inputCol='review', outputCol=('token_text'))
stop_remove = StopWordsRemover(inputCol='token_text', outputCol='stop_token')
count_vec = CountVectorizer(inputCol='stop_token',outputCol='CountVect')
idf = IDF(inputCol='CountVect',outputCol='features')
data_prepare = Pipeline(stages=[tokenizer, stop_remove, count_vec,idf])
cleaner = data_prepare.fit(dataset_refined)
clean_data = cleaner.transform(dataset_refined)
clean_data = clean_data.select('label','features','review').dropna()
training,test = clean_data.randomSplit([0.7,0.3])
training = training.dropna()
get_naive = NaiveBayes()
model = get_naive.fit(training)
test_results = model.transform(test)
test_results.show()
test_results.select('label','review','prediction').write.csv('/home/cdh@psnet.com/Gourav/sentiments/')
Code to evaluate the performance of trained model on a testing dataset
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
get_eval= MulticlassClassificationEvaluator()
get_eval = get_eval.evaluate(test_results)
print(get_eval)

#Code to show the implementation of Logistic Regression for predicting the sentiments of sentences. This code will be stitched after splitting the training and testing dataset in the above code.

from pyspark.ml.classification import LogisticRegression
my_model = LogisticRegression()
fitted_lg = my_model.fit(training)
log_summary = fitted_lg.summary
log_summary.predictions.show()
predictions = fitted_lg.evaluate(test)
my_eval = BinaryClassificationEvaluator()
test_result = my_eval.evaluate(predictions.predictions)
test_result
