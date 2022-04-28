from pyspark import SparkContext 
sc = SparkContext(“local”, “Broadcast”) 
words_new = sc.broadcast(["Big Data", "Machine learning", "Analytics", "Deep Learning", "Artificial Intelligence"]) 
data = words_new.value 
print "Stored data -> %s" % (data) 
elem = words_new.value[2] 
print "Printing a particular element in RDD -> %s" % (elem)
