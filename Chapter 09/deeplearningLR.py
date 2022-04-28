import pandas as pd 
from tensorflow import keras
from tensorflow.keras import layers
import pyspark
from pyspark import SparkContext, SparkConf
from elephas.utils.rdd_utils import to_simple_rdd
from sklearn.metrics import confusion_matrix
from elephas.spark_model import SparkModel
from elephas.utils.rdd_utils import to_simple_rdd
from sklearn.model_selection import train_test_split
import elephas
import pyspark
import tensorflow as tf
import keras
from keras import layers
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
conf = SparkConf().setAppName('distributed-framework-Elephas').setMaster('local[9]')
sc = SparkContext(conf=conf)
dataset = pd.read_csv('/content/drive/MyDrive/Salary_Data.csv')
X = dataset.iloc[:, :-1].values
print(X)
y = dataset.iloc[:, -1].values
print(y)
# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/4, random_state = 0)
model = keras.Sequential()
model.add(layers.Dense(128, activation="relu", input_dim=1))#, input_dim=1))
model.add(layers.Dense(128, activation="relu"))
model.add(layers.Dense(128, activation="relu"))
model.add(layers.Dense(64, activation="relu"))
model.add(layers.Dense(64, activation="relu"))
model.add(layers.Dense(32, activation="relu"))
model.add(layers.Dense(32, activation="relu"))
model.add(layers.Dense(1))
model.compile(optimizer="adam", loss="mse", metrics=["mae"])
model.sumarry()
rdd = to_simple_rdd(sc, X_train, y_train)
spark_model = SparkModel(model, frequency='epoch', mode='asynchronous')
spark_model.fit(rdd, epochs=20, batch_size=32, verbose=0, validation_split=0.1)
spark_model.save('/content/drive/MyDrive/')
predictions = spark_model.predict(X_test)
score = spark_model.master_network.evaluate(X_test, y_test, verbose=2)
print('Test accuracy: ', score[1]/1000)
