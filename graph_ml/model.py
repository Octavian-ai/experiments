
import keras
from keras.models import Sequential
from keras.layers import *

class Model(object):

	@staticmethod
	def generate(params, dataset):

		if params.experiment == "review_from_visible_style":

			model = Sequential([
				Dense(8, 
					input_shape=dataset.input_shape,
					activation='softmax'),
				Dense(1, activation='sigmoid'),
			])

			model.compile(loss=keras.losses.mean_squared_error,
				optimizer=keras.optimizers.SGD(lr=0.3),
				metrics=['accuracy'])

		return model