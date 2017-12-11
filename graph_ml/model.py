
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

class Model(object):

	@staticmethod
	def generate(params, dataset):

		if params.experiment == "simple":

			model = Sequential()
			model.add(Dense(10, input_shape=dataset.input_shape, activation='relu'))
			model.add(Dropout(0.5))
			model.add(Dense(1, activation='relu'))

			model.compile(loss=keras.losses.mean_squared_error,
				optimizer=keras.optimizers.Adadelta(),
				metrics=['accuracy'])

			return model