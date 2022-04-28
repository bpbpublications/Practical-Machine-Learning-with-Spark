from pyspark.mllib.linalg import Matrix, Matrices
get_dense_matrix = Matrices.dense(2, 3, [1, 3, 5, 2, 4, 6])
print(get_dense_matrix.toArray())
