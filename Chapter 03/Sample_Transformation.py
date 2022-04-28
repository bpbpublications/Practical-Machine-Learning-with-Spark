get_rdd = sc.parallelize(['This','book','will','help','all','the','Big','Data','and','Machine','Learning','aspirants'])
get_rdd.collect()
['This', 'book', 'will', 'help', 'all', 'the', 'Big', 'Data', 'and', 'Machine', 'Learning', 'aspirants']
print(type(get_rdd))
<class 'pyspark.rdd.RDD'>
get_sampled = get_rdd.sample(False, 0.6)
get_sampled.collect()
