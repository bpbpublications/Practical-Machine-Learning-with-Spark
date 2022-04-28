from pyspark.mllib.linalg.distributed import RowMatrix
rowsRDD = sc.parallelize([[11,12], [22, 33], [33, 55], [19, 18]])
get_distributed_mat = RowMatrix(rowsRDD)
print(get_distributed_mat)
print(type(get_distributed_mat))
m_rows = get_distributed_mat.numRows()
m_rows  
n_cols = get_distributed_mat.numCols() 
n_cols

