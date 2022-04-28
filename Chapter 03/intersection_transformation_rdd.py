RDD1 = sc.parallelize(range(1,10))
RDD2 = sc.parallelize(range(5,15))
RDD1.intersection(RDD2).collect()
[5, 6, 7, 8, 9]
