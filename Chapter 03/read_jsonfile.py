This program depicts the way to read the JSON data using PySpark framework.
from pyspark.sql import SparkSession
from pyspark.sql import Row
spark = SparkSession.builder.appName("JSON INTEGRATION").getOrCreate()
df = spark.read.option("multiline","true").json("Gourav/chap3/total-pounds-of-food-produced-locally-96-17-json.json")
df.show()
