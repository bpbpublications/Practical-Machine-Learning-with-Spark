DD1 = sc.parallelize(range(1,10))
RDD2 = sc.parallelize(range(10,21))
RDD1.union(RDD2).collect()
