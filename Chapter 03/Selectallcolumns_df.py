all_columns =dataframe.select([col(c) for c in dataframe.columns])
all_columns.show()
