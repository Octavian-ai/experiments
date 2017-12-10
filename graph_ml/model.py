
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

class Model(object):

	@staticmethod
	def generate(params, dataset):

		if params.experiment == "simple":
			num_classes = 2

			model = Sequential()
			model.add(Dense(input_shape=dataset.input_shape, 10, activation='relu'))
			model.add(Dropout(0.5))
			model.add(Dense(num_classes, activation='softmax'))

			return model