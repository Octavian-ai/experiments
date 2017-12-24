
import keras
import keras.backend as K

from keras.models import Model
from keras.layers import *
from recurrentshop.engine import RNNCell


class PatchCell(RNNCell):

	def __init__(self, experiment, **kwargs):
		self.experiment = experiment
		self.word_size = self.experiment.header.meta["word_size"]
		self.batch_size = self.experiment.params.batch_size
		self.memory_size = self.experiment.header.meta["memory_size"]

		self.word_shape = [self.word_size]
		self.word_shape_batch = [self.batch_size, self.word_size]
		self.memory_shape = [self.memory_size, self.word_size]
		self.memory_shape_batch = [self.batch_size] + self.memory_shape

		super(PatchCell, self).__init__(**kwargs)

	def expand_word_mask(self, address, mask):
		# if len(mask.shape) == 2:
		mask = K.expand_dims(mask,1)
		address = K.expand_dims(address, -1)

		mask = K.batch_dot(address, mask)
		assert mask.shape == self.memory_shape_batch, (f"Mask is not memory shaped {mask.shape}")
		return mask

	def read(self, memory, address):
		address_expanded = K.repeat_elements(K.expand_dims(address, -1), self.word_size, -1)
		read_e = memory * address_expanded
		return K.sum(read_e, axis=1)

	def write(self, memory, address, write):
		write_e = self.expand_word_mask(address, write)
		return memory + write_e

	def erase(self, memory, address, erase):
		erase_e = self.expand_word_mask(address, erase)

		# I need to make this 
		# self.memory_ones = self.add_weight(shape=self.memory_shape_batch,
		# 	initializer='ones',
		# 	name='memory_ones')

		# memory_ones = K.zeros(self.memory_shape_batch) #  Initializer for variable recurrent_model_1/while/model_1/lambda_3/Variable/ is from inside a control-flow construct, such as a loop or conditional. When creating a variable inside a loop or conditional, use a lambda as the initializer.
		memory_ones = K.constant(0, shape=self.memory_shape_batch)

		# memory_ones = tf.get_variable("memory_ones", shape=self.memory_shape_batch, initializer=tf.constant_inititializer(0))

		memory_out = memory * (memory_ones - erase_e)
		return memory_out

	def memory_op(self, memory, address, write, erase):

		# Python lacks currying
		read_layer  = Lambda(lambda x:  self.read(x, address), output_shape=self.word_shape)
		write_layer = Lambda(lambda x: self.write(x, address, write), output_shape=self.memory_shape)
		erase_layer = Lambda(lambda x: self.erase(x, address, erase), output_shape=self.memory_shape)

		output = read_layer(memory)
		memory = write_layer(memory)
		memory = erase_layer(memory)

		assert output.shape == (self.word_shape_batch), f"Output is not a memory word {output.shape}"

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

