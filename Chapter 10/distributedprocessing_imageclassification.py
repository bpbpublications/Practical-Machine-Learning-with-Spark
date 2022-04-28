>>%pip install elephas==0.4.3
>>%pip install tensorflow==1.14.0
>>%pip install keras==2.2.0
>>import matplotlib.pyplot as plt
>>from keras.models import Sequential
>>from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
>>from elephas.spark_model import SparkModel
>>from elephas.utils.rdd_utils import to_simple_rdd
>>from pyspark import SparkContext, SparkConf
>>from keras.utils import np_utils
>>import keras
>>import cv2
>>from google.colab.patches import cv2_imshow
>>from keras import optimizers
>>from pyspark.sql.functions import rand
>>from pyspark.mllib.evaluation import MulticlassMetrics
>>from elephas.ml_model import ElephasEstimator
>>from keras.datasets import fashion_mnist
>>from tensorflow.keras.utils import to_categorical
>>conf = SparkConf().setAppName('distributed-framework-Elephas').setMaster('local')
>>sc = SparkContext(conf=conf)
>>(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
>>x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
>>x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
>>x_train = x_train.astype('float32')/255
>>x_test = x_test.astype('float32')/255
>>y_train = keras.utils.to_categorical(y_train, 10)
>>y_test = keras.utils.to_categorical(y_test, 10)
>>model = Sequential()
>>model.add(Conv2D(28, kernel_size=(3,3), input_shape= (28,28,1), name="convlayer1"))
>>model.add(MaxPooling2D(pool_size=(2, 2)))
>>model.add(Conv2D(28, kernel_size=(3,3), name="convlayer2"))
>>model.add(MaxPooling2D(pool_size=(2, 2)))
>>model.add(Flatten())
>>model.add(Dense(128, activation="relu",name='fclayer1'))
>>model.add(Dropout(0.2))
>>model.add(Dense(10,activation='softmax', name="output"))
>>model.compile(optimizer='adam', loss="categorical_crossentropy", metrics=['accuracy'])
>>_create_rdd = to_simple_rdd(sc, x_train, y_train)
>>spark_model = SparkModel(model, frequency="epoch", mode="asynchronous")
>>spark_model.fit(_create_rdd, epochs=10, batch_size=128, verbose=1,validation_split=0.3)
>>model.layers
>>post_image_index = 100
>>for index, get_image_id in enumerate(range(100)):
 	 plt.imshow(x_test[get_image_id].reshape(28, 28),cmap='viridis')
  	pred = spark_model.predict(x_test[get_image_id].reshape(1, 28, 28, 1))
 	get_pred = str(pred.argmax())
>>get_prediction = spark_model.master_network.evaluate(x_test, y_test, verbose=2)
>>print(get_prediction[1]*100)
