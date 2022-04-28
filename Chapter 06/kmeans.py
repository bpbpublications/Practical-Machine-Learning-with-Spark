!pip install pyspark==2.1.2
import pyspark
conf = pyspark.SparkConf()
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pyspark.sql import SparkSession
from pyspark.ml.feature import StandardScaler
from pyspark.ml.clustering import KMeans,KMeansModel,KMeansSummary
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
spark = SparkSession.builder.appName('KMeans').getOrCreate()
df = spark.read.format('com.databricks.spark.csv').options(header='true', inferschema='true').load('/content/sample_data/Mall_Customers.csv')
assembler = VectorAssembler(inputCols = ["Age", "Annual_Income","SpendingScore"], outputCol = "features")
output = assembler.transform(df)
output.show(5)
scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures")
scaler_model = scaler.fit(output)
print(scaler_model)
final_data = scaler_model.transform(output)
final_data.show(5)
#Clustering
KMeans = KMeans(featuresCol="scaledFeatures", k=4, predictionCol="prediction")
model = KMeans.fit(final_data)
cost = model.computeCost(final_data)
print("Within Set Sum of Squared Errors = ",cost)
centers = model.clusterCenters()
print("Computing Cluster Centers")
for center in centers:
print(center)
model = model.transform(final_data)
model.show(5)
#Converting into Pandas
model = model.toPandas()
#Scatter Plot 
sc_plt = plt.figure(figsize=(17,12)).gca(projection='3d')
sc_plt.scatter(model.Age,model.SpendingScore, model.Annual_Income, c=model.prediction)
sc_plt.set_xlabel('x')
sc_plt.set_ylabel('y')
sc_plt.set_zlabel('z')
plt.show()

