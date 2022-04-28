from pyspark.sql import SparkSession
from pyspark.sql import Row
spark = SparkSession.builder.appName("ORC-PYSPARK BRIDGE").getOrCreate()
read_ORC= spark.read.option("header","true").orc("/home /Gourav/chap3/userdata1_orc")
#Display content of table
read_ORC.show(5)
#Getting Datatype information of table
read_ORC.printSchema()
