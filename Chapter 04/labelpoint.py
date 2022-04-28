from pyspark.mllib.regression import LabeledPoint
get_densevec = Vectors.dense([1,2,3,4,5])
get_labeled_point = LabeledPoint(2,get_densevec)
# To display the Features 
print(get_labeled_point.features)
# To display the Label
print(get_labeled_point.label)
