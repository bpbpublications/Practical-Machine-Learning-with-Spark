from pyspark.sql import SparkSession
from pyspark.sql import Row
spark = SparkSession.builder.appName("Python Spark SQL Hive integration example").config("hive.metastore.uris", "thrift://*******:9083").enableHiveSupport().getOrCreate()
spark.sql('show tables').show()
