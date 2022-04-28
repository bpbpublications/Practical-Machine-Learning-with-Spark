from pyspark import SparkContext
from pyspark.sql import SQLContext
sc = SparkContext(appName="MariaDB-PySpark Bridge")
sqlContext = SQLContext(sc)
source_df = sqlContext.read.format('jdbc').options(
          url='jdbc:mysql://localhost/test',
          driver='com.mysql.jdbc.Driver',
          dbtable='processed_data',
          user='cdh').load()
source_df.show(3)
