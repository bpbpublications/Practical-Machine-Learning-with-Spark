from pyspark.sql import SparkSession
from pyspark.sql import Row
spark = SparkSession.builder.appName("AWS S3-PYSPARK BRIDGE1").getOrCreate()
get_s3= spark.read.parquet("s3://path of bucket")
#Display content of table
get_s3.show(10)
