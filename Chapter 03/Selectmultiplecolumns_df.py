multiple_columns = dataframe.select("Department","Wage").show(truncate=False)
or
from pyspark.sql.functions import col
dataframe.select(col("Department"),col("Age")).show()
