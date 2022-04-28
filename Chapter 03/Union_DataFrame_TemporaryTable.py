df1.createOrReplaceTempView("df1")
df1.createOrReplaceTempView("df2")
df3 = df2.union(df1)
df3.createOrReplaceTempView("df3")
df4 = spark.sql("select Item_ID, Item_Name, sum(Quantity) as Quantity from df3 group by Item_ID, Item_Name")
df4.show(10)
