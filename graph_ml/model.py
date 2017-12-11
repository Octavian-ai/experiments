
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

class Model(object):

	@staticmethod
	def generate(params, dataset):

		if params.experiment == "simple":

			model = Sequential()
			model.add(Dense(4, input_shape=dataset.input_shape, activation='relu'))
			model.add(Dropout(0.01))
			model.add(Dense(1))

			model.compile(loss=keras.losses.mean_squared_error,
				optimizer=keras.optimizers.Adam(),
				metrics=['accuracy'])

		return model