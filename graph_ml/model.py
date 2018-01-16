
import keras
from keras.models import Sequential, Model
from keras.layers import *
import keras.backend as K

import tensorflow as tf

from .ntm import *
from .adjacency_layer import Adjacency


# Rainbow sprinkles for your activation function
# Try to use all activation functions
# @argument m: (?,N) tensor
# @returns (?,N*5) tensor
def PolyActivation(m):
	# wildcard of the day - let's do inception style activation because I've no idea which is best
	# and frequently I get great boosts from switching activation functions
	activations = ['tanh', 'sigmoid', 'softmax', 'softplus', 'relu']

	# TODO: Add dense layer to resize back to original size
	# I cannot work out how to do that in Keras yet :/
	return Concatenate()([
		Activation(i)(m) for i in activations
	])


# Choose activation function for me
# More efficient than PolyActivation
# @returns Same sized tensor as input
def PolySwitchActivation(m):
	# will fail for shared nodes
	print(m.shape)

	if len(m.shape) != 3:
		# TODO: make this work in a sane way
		m = Reshape([i for i in m.shape.dims if i is not None] + [1])(m) # warning: assumes tensorflow

	activations = ['tanh', 'sigmoid', 'softmax', 'softplus', 'relu']
	return add([
		Conv1D(1,1)(Activation(i)(m)) for i in activations
	])

class Model(object):

	@classmethod
	def generate(cls, experiment, dataset):
		params = experiment.params

		# TODO: Move this into Experiment header
		n_styles = 6
		n_sequence = 100

		bs = experiment.params.batch_size

		if experiment.name == "review_from_visible_style":
			model = Sequential([
				Dense(8, 
					input_shape=dataset.input_shape,
					activation='softmax'),
				Dense(1, activation='sigmoid'),
			])


		elif experiment.name == "review_from_hidden_style_neighbor_conv":
			neighbors = Input(shape=(n_sequence,n_styles+2,), dtype='float32', name='neighbors')
			person = Input(shape=(n_styles,), dtype='float32', name='person')

			m = cls.style_from_neighbors(neighbors, n_styles, n_sequence)
			m = Concatenate()([m, person])
			m = Dense(n_styles*4)(m)
			m = PolyActivation(m)
			m = Dense(1, activation='sigmoid')(m)

			model = keras.models.Model(inputs=[person, neighbors], outputs=[m])

		
		elif experiment.name == "style_from_neighbor_conv":
			neighbors = Input(shape=(n_sequence,n_styles+2,), dtype='float32', name='neighbors')
			m = cls.style_from_neighbors(neighbors, n_styles, n_sequence)

			model = keras.models.Model(inputs=[neighbors], outputs=[m])


		elif experiment.name == "style_from_neighbor_rnn":
			neighbors = Input(shape=(n_sequence,n_styles+2,), dtype='float32', name='neighbors')
			m = LSTM(n_styles*4)(neighbors)
			m = Dense(n_styles)(m)
			m = Activation('sigmoid', name='final_activation')(m)

			model = keras.models.Model(inputs=[neighbors], outputs=[m])


		elif experiment.name == "review_from_all_hidden_simple_unroll":
			thinking_width = 10

			neighbors = Input(shape=(experiment.header.params["neighbor_count"],4,), dtype='float32', name='neighbors')
			m = Conv1D(thinking_width, 1, activation='tanh')(neighbors)
			m = MaxPooling1D(experiment.header.params["neighbor_count"])(m)
			m = Reshape([thinking_width])(m)
			m = Dense(1)(m)
			m = Activation("sigmoid", name='final_activation')(m)

			model = keras.models.Model(inputs=[neighbors], outputs=[m])


		elif experiment.name == 'review_from_all_hidden_random_walks':

			ss = experiment.header.params["sequence_size"]
			ps = experiment.header.params["patch_size"]
			pw = experiment.header.params["patch_width"]

			patch = Input(batch_shape=(bs,ss,ps,pw), dtype='float32', name="patch")
			# flat_patch = Reshape([ss*ps*pw])(patch)
			# score = Dense(experiment.header.params["working_width"]*2, activation="tanh")(flat_patch)
			# score = Dense(experiment.header.params["working_width"],   activation="tanh")(flat_patch)

			# rnn = PatchNTM(experiment).build()
			# score = rnn(patch)

			# Data format
			# x      = [x_path, x_path, x_path]
			# x_path = [x_node, x_node, x_node]
			# x_node = [label, score, is_head]

			# x = [
			# 	[
			# 		[label, score, is_head]:Node, 
			# 		[label, score, is_head]:Node
			# 	]:Path, 
			# 	[
			# 		[label, score, is_head]:Node, 
			# 		[label, score, is_head]:Node
			# 	]:Path 
			# ]:Sequence

			# Convolve path-pattern
			channels = 8
			pattern_length = 8

			m = patch

			# Add channels for convolution
			m = Lambda(lambda x: K.expand_dims(x, axis=-1))(m)

			# Compute!!
			m = Conv3D(channels, (1, pattern_length, pw), activation='relu')(m)
			pattern_conv_out_size = ps - pattern_length + 1

			m = Reshape([ss * channels * pattern_conv_out_size])(m)
			m = Dense(4, activation="relu", name="score_dense")(m)
			score = Dense(1, activation="sigmoid", name="score_out")(m)
			
			model = keras.models.Model(inputs=[patch], outputs=[score])


		elif experiment.name == 'review_from_all_hidden_adj':

			pr_c = experiment.header.params["product_count"]
			pe_c = experiment.header.params["person_count"]
			style_width = experiment.header.params["style_width"]

			adj_con = Input(batch_shape=(bs, pr_c, pe_c), dtype='float32', name="adj_con")
			scores = Adjacency(pe_c, pr_c, style_width, name="hidden_to_adj")(adj_con)

			model = keras.models.Model(inputs=[adj_con], outputs=[scores])

			model.compile(loss=keras.losses.mean_squared_error,
				optimizer=keras.optimizers.SGD(lr=5000),
				metrics=['accuracy'])

			return model



		# Compile time!
		if experiment.header.target == float:
			model.compile(loss=keras.losses.mean_squared_error,
				optimizer=keras.optimizers.SGD(lr=0.3),
				metrics=['accuracy'])

		elif experiment.header.target == list:
			model.compile(loss='categorical_crossentropy',
				optimizer=keras.optimizers.SGD(lr=0.3),
				metrics=['accuracy'])



		return model

	@classmethod 
	def style_from_neighbors(cls, neighbors, n_styles, n_sequence):
		m = Conv1D(n_styles, 1, activation='tanh')(neighbors)
		m = MaxPooling1D(n_sequence)(m)
		m = Reshape([n_styles])(m)
		m = Dense(n_styles)(m)
		m = Activation('softmax')(m)

		return m


