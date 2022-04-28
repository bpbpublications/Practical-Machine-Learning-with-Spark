from pyspark.ml.feature import PCA
from pyspark.ml.linalg import Vectors
dataset =  [(Vectors.dense([2.0, 0.0, 3.0, 4.0, 5.0]),),
        (Vectors.dense([4.0, 0.0, 0.0, 6.0, 7.0]),)]
df_created = spark.createDataFrame(dataset, ["vector_space"])
get_pca = PCA(k=2, inputCol="vector_space", outputCol="PCA_Outcome")
train_model = get_pca.fit(df_created)
model_result = train_model.transform(df_created).select("PCA_Outcome")
model_result.show(truncate=False)
