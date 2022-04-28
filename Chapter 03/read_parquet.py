from pyspark.sql import SparkSession
from pyspark.sql import Row
spark = SparkSession.builder.appName("PARQUET-PYSPARK BRIDGE1").getOrCreate()
get_parquet= spark.read.parquet("/home /Gourav/chap3/userdata1.parquet")
#Display content of table
get_parquet.show(10)
#Getting Datatype information of table
get_parquet.printSchema()
#Registering into a temporary table
get_parquet.registerTempTable("parquet_table")
#Group By transformation on country column
get_transformation = spark.sql("SELECT country,count(1) as count FROM parquet_table GROUP BY country")
#Write into the directory after the transformation
get_transformation.write.mode('overwrite').parquet("Sales.parquet")

