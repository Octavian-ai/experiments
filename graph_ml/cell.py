
import keras
import keras.backend as K

from keras.models import Model
from keras.layers import *
from recurrentshop.engine import RNNCell


class PatchCell(RNNCell):

	def __init__(self, experiment, **kwargs):
		self.experiment = experiment
		self.word_size = self.experiment.header.meta["word_size"]
		super(PatchCell, self).__init__(**kwargs)


	def read(self, memory, address):
		masked = K.batch_dot(memory, address)
		row = K.sum(masked, axis=-1)
		return row

	def write(self, memory, address, write):
		return memory + K.dot(address, write)

	def erase(self, memory, address, erase):
		erase = K.expand_dims(erase,1)
		erase_expand = K.batch_dot(address, erase)
		erase_block = (K.ones((self.batch_size, self.memory_size, self.word_size)) - erase_expand)
		memory_out = K.dot(memory, erase_block)
		return memory_out

	def memory_op(self, memory, address, write, erase):
		address_expanded = K.repeat_elements(K.expand_dims(address, 1), self.word_size, 1)

		read_layer = Lambda(lambda x: K.sum(K.batch_dot(memory, x), axis=-1))

		# output = self.read(memory, address_expanded)
		output = read_layer(address_expanded)
		# memory = self.erase(memory, address, erase)
		# memory = self.write(memory, address, write)

		return output, memory


	def build_model(self, input_shape):
		batch_size  = self.experiment.params.batch_size
		patch_size  = self.experiment.header.meta["patch_size"]
		patch_width = self.experiment.header.meta["patch_width"]
		memory_size = self.experiment.header.meta["memory_size"]
		word_size   = self.experiment.header.meta["word_size"]
		node_control_width = self.experiment.header.meta["node_control_width"]

		patch = Input(batch_shape=(batch_size, patch_size, patch_width))
		memory_in = Input(batch_shape=(batch_size, memory_size, word_size))

		assert patch.shape == (batch_size, patch_size, patch_width)

		n = Conv1D(
			filters=node_control_width, 
			kernel_size=1, 
			activation='tanh', 
			kernel_initializer='random_uniform',
			bias_initializer='zeros')(patch)

		n = MaxPooling1D(patch_size)(n)
		n = Reshape([node_control_width])(n)

		all_control = n

		address_reference = Dense(patch_size)(all_control)
		write = Dense(word_size)(all_control)
		erase = Dense(word_size)(all_control)

		address_resolved = NodeAddressor(batch_size, memory_size)(address_reference, patch)

		assert memory_in.shape 		  == (batch_size, memory_size, word_size)
		assert address_resolved.shape == (batch_size, memory_size)
		assert erase.shape   		  == (batch_size, word_size)
		assert write.shape     		  == (batch_size, word_size)

		out, memory_out = self.memory_op(memory_in, address_resolved, write, erase)

		# out = Concatenate()([all_control, out])
		out = Dense(patch_width)(out)

		return Model([patch, memory_in], [address_reference, memory_out])



class PatchRNN(object):

	def __init__(self, experiment):
		self.experiment = experiment

	def __call__(self, layer_in):
		patch_size  = self.experiment.header.meta["patch_size"]
		patch_width = self.experiment.header.meta["patch_width"]
		cell = PatchCell(self.experiment, output_dim=patch_width, input_shape=(patch_size, patch_width))
		# get_layer accepts arguments like return_sequences, unroll etc :
		return cell.get_layer(return_sequences=True)(layer_in)


class NodeAddressor(object):

	def __init__(self, batch_size, memory_size):
		self.batch_size = batch_size
		self.memory_size = memory_size

	def __call__(self, address, patch):
		return K.ones((self.batch_size, self.memory_size))

