
import keras
from keras.models import Sequential, Model
from keras.layers import *

class Model(object):

	@classmethod
	def generate(cls, params, dataset):

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

			m = cls.style_from_neighbors(neighbors, n_styles)
			m = Concatenate()([m, person])
			m = Dense(n_styles*n_styles, activation='softmax')(m)
			m = Dense(1, activation='sigmoid')(m)

			model = keras.models.Model(inputs=[person, neighbors], outputs=[m])

			model.compile(loss=keras.losses.mean_squared_error,
				optimizer=keras.optimizers.SGD(lr=0.3),
				metrics=['accuracy'])

		elif params.experiment == "style_from_neighbor_conv":

			n_styles = 6

			neighbors = Input(shape=(100,n_styles+2,), dtype='float32', name='neighbors')
			m = cls.style_from_neighbors(neighbors, n_styles)

			model = keras.models.Model(inputs=[neighbors], outputs=[m])

			model.compile(loss='categorical_crossentropy',
				optimizer=keras.optimizers.SGD(lr=0.3),
				metrics=['accuracy'])

		return model

	@classmethod 
	def style_from_neighbors(cls, neighbors, n_styles):
		m = Conv1D(n_styles, 1, activation='tanh')(neighbors)
		m = MaxPooling1D(100)(m)
		m = Reshape([n_styles])(m)
		m = Dense(n_styles)(m)
		m = Activation('softmax')(m)

		return m


