
import keras
from keras.models import Sequential, Model
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

		elif params.experiment == "review_from_hidden_style_neighbor_conv":

			n_styles = 6

			neighbors = Input(shape=(100,n_styles+2,), dtype='float32', name='neighbors')
			person = Input(shape=(n_styles,), dtype='float32', name='person')

			m = Conv1D(n_styles*2, 1, activation='softmax')(neighbors)
			m = AveragePooling1D(100)(m)
			m = Reshape([n_styles*2])(m)
			m = Concatenate()([m, person])
			m = Dense(n_styles*n_styles, activation='softmax')(m)
			m = Dense(1, activation='sigmoid')(m)
			score = m

			model = keras.models.Model(inputs=[person, neighbors], outputs=[score])

			model.compile(loss=keras.losses.mean_squared_error,
				optimizer=keras.optimizers.SGD(lr=0.3),
				metrics=['accuracy'])

		return model

