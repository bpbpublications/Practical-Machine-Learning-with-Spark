get_flatmap_transform = df.select(df,columns[0]).rdd.flatMap(lambda x: (x,1))
get_map_transform.take(10)
