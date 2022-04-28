%pip install pyspark==3.1.1
from pyspark.ml.classification import LogisticRegression, OneVsRest
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql import SparkSession
from pyspark.sql import SQLContext
from pyspark.ml.feature import VectorAssembler
#Spark's application and loading of dataset
spark = SparkSession.builder.appName('Gradient Boosted Tree Classifier').getOrCreate()
data = spark.read.csv(‘/content/sample_data/cancerdata.csv’,inferSchema=True, header=True)
#Dropping the rows that contains “Null Value”
data = data.dropna()
#Show the columns of dataframe
data.columns
#Converting the datatype into double or int
converteddata = data.selectExpr("cast(age as int) age",
    "cast(education as int) education",
    "cast(currentSmoker as double) currentSmoker",
    "cast(TenYearCHD as double) TenYearCHD",
	"cast(male as int) gender",
	"cast(cigsPerDay as double) cigsPerDay",
	"cast(BPMeds as double) BPMeds",
	"cast(prevalentStroke as double) prevalentStroke",
	"cast(prevalentHyp as double) prevalentHyp",
	"cast(diabetes as double) diabetes",
	"cast(totChol as double) totChol",
	"cast(sysBP as double) sysBP",
	"cast(diaBP as double) diaBP",
	"cast(BMI as double) BMI",
	"cast(heartRate as double) heartRate",
	"cast(glucose as double) glucose")
#Key features selection and converting into vectors
converteddata.dropna()
assembler = VectorAssembler(inputCols=['age', 'education', 'currentSmoker', 'gender','glucose','diabetes'],outputCol='features')
get_output = assembler.transform(converteddata)
get_output.printSchema()
finalized_data = get_output.select("features", "TenYearCHD")
finalized_data = finalized_data.selectExpr("features", "TenYearCHD as label")
training_data, testing_data = finalized_data.randomSplit([0.7,0.3])
# Initializing the classifier base
lr_base = LogisticRegression(maxIter=100, tol=1E-6, fitIntercept=True, elasticNetParam=0.6)
# Initializing the One Vs Rest Classifier
ovr_base = OneVsRest(classifier=lr_base,featuresCol='features', labelCol='label')
print(type(ovr_base))
# train the multiclass model.
trained_model_ovr = ovr.fit(training_data)
# tranforming operation on testing dataset
get_predictions = trained_model_ovr.transform(testing_data)
get_data = get_predictions.select("features","label", "prediction")
get_data.show(5)
# Evaluating the model
evaluator = MulticlassClassificationEvaluator(metricName="accuracy")
# Accuracy calculation on testing dataset
accuracy = evaluator.evaluate(get_predictions)
print("Test Error = %g",(accuracy))
