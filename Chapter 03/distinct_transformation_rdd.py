RDD1 = sc.parallelize(range(1,13))
RDD2 = sc.parallelize(range(7,20))
RDD1.union(RDD2).distinct().collect()
