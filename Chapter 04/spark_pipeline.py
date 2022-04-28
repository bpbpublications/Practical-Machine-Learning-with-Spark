from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorIndexer, VectorAssembler
create_df = spark.createDataFrame([
    (0, Vectors.dense([3.0, 6.0, -4.0]), 18.0),
    (1, Vectors.dense([3.0, 2.0, 10.0]), 30.0)
], ["unique_id", "get_features", "user_age"])
vector_indexer = VectorIndexer(inputCol="get_features", outputCol="get_result")
assembler = VectorAssembler(inputCols=["unique_id","get_features","get_result"], outputCol="get_output")
pipeline = Pipeline(stages=[vector_indexer, assembler])
model = pipeline.fit(create_df).transform(create_df)
model.show()
