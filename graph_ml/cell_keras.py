
import keras
import keras.backend as K
from keras.layers import *

class PatchRNN(keras.layers.RNN):
	def __init__(self, input_dim=None, **kwargs):
		super(PatchRNN, self).__init__(**kwargs)

		if input_dim is not None:
			self.input_spec = [InputSpec(ndim=input_dim)]

class AddressableCell(keras.layers.Layer):

	# TODO: share memory across batch
	def __init__(self, experiment, **kwargs):
		self.batch_size = experiment.params.batch_size
		self.memory_size = experiment.header.meta["memory_size"]
		self.word_size = experiment.header.meta["word_size"]
		self.state_size = (self.memory_size * self.word_size)

		super(AddressableCell, self).__init__(**kwargs)

	def build(self, input_shape):
		self.built = True


	# TODO: work out actual fns
	def read(self, memory, address):
		masked = K.batch_dot(memory, address)
		row = K.sum(masked, axis=-1)
		return row

	def write(self, memory, address, write):
		return memory + K.dot(address, write)

	def erase(self, memory, address, erase):
		erase = K.expand_dims(erase,1)

		print("erase", erase)
		erase_expand = K.batch_dot(address, erase)
		print("erase_expand", erase_expand)
		erase_block = (K.ones((self.batch_size, self.memory_size, self.word_size)) - erase_expand)
		print("erase_block", erase_block)
		memory_out = K.dot(memory, erase_block)
		print("memory_out", memory_out)
		return memory_out
	# -----------------------------

	def call(self, inputs, states):
		address, write, erase = inputs

		memory = Reshape([self.memory_size,self.word_size])(states[0])

		address_expanded = K.repeat_elements(K.expand_dims(address, 1), self.word_size, 1)

		assert memory.shape  == (self.batch_size, self.memory_size, self.word_size)
		assert address.shape == (self.batch_size, self.memory_size)
		assert address_expanded.shape == (self.batch_size, self.word_size, self.memory_size), f"Address expanded wrong shape {address_expanded.shape}"
		assert erase.shape   == (self.batch_size, self.word_size)
		assert write.shape   == (self.batch_size, self.word_size)

		output = self.read(memory, address_expanded)
		# memory = self.erase(memory, address, erase)
		# memory = self.write(memory, address, write)

		memory_flat = Reshape([self.memory_size*self.word_size])(memory)

		return output, [memory_flat]

class PatchCell(AddressableCell):

	def __init__(self, **kwargs):
		super(PatchCell, self).__init__(**kwargs)

	def build(self, input_shape):
		super(PatchCell, self).build(self, input_shape)

		self.conv_kernel = self.add_weight(shape=(input_shape[-1], self.units),
			initializer='uniform',
			name='kernel')

		self.build = True

	def call(self, inputs, states):

		node_control_width = 16
		neighbor_count = 20
		node_width = 5
		patch_length = neighbor_count+1

		# This was nicer but I've yet to hack RNN to allow it
		# node, neighbors = inputs
		flat_patch = inputs

		# boring ugh
		patch = Reshape([patch_length, node_width])(flat_patch)



		m = Conv1D(
			filters=node_control_width, 
			kernel_size=1, 
			activation='tanh', 
			kernel_initializer='random_uniform',
			bias_initializer='zeros')(patch)

		m = MaxPooling1D(patch_length)(m)
		m = Reshape([node_control_width])(m)

		all_control = m

		address_reference = Dense(patch_length)(all_control)
		write = Dense(self.word_size)(all_control)
		erase = Dense(self.word_size)(all_control)

		# Nodes store one-hot encoding of their memory location
		# address is relative to the nodes in this patch
		# Take address and transpose it, then multiply that by the
		# N x M matrix of this patches one-hot node locations

		address_resolved = NodeAddressor(self.batch_size, self.memory_size)(address_reference, patch)

		c_out, c_state = AddressableCell.call(self, (address_resolved, write, erase), states)

		out = Concatenate()([all_control, c_out])
		
		return out, [c_state]



class NodeAddressor(object):

	def __init__(self, batch_size, memory_size):
		self.batch_size = batch_size
		self.memory_size = memory_size

	def __call__(self, address, patch):
		return K.ones((self.batch_size, self.memory_size))






