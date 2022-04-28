get_rdd = sc.parallelize(range(1,5000))
get_rdd.sum()
