
import keras
import keras.backend as K

import tensorflow as tf

from keras.models import Model
from keras.layers import *
from recurrentshop import RecurrentModel

from .util import *

class PatchBase(object):

	def __init__(self, experiment):
		self.experiment = experiment

		self.patch_size  = experiment.header.params["patch_size"]
		self.patch_width = experiment.header.params["patch_width"]

		self.word_size = self.experiment.header.params["word_size"]
		self.batch_size = self.experiment.params.batch_size
		self.memory_size = self.experiment.header.params["memory_size"]

		self.word_shape = [self.word_size]
		self.word_shape_batch = [self.batch_size, self.word_size]
		self.memory_shape = [self.memory_size, self.word_size]
		self.memory_shape_batch = [None] + self.memory_shape

	def combine_nodes(self, patch, width):
		n1 = Conv1D(
			filters=width, 
			kernel_size=1, 
			activation='tanh', 
			kernel_initializer='random_uniform',
			bias_initializer='zeros',
			name="ConvPatch1")(patch)

		n2 = Conv1D(
			filters=width, 
			kernel_size=1, 
			activation='tanh', 
			kernel_initializer='random_uniform',
			bias_initializer='zeros',
			name="ConvPatch2")(patch)

		n = multiply([n1, n2])

		n = MaxPooling1D(self.patch_size)(n)
		n = Reshape([width])(n)
		return n

	def patch_extract(self, address, patch, slice_begin):
		extract_width = self.patch_width - (slice_begin % self.patch_width)

		address_repeated = Lambda(lambda x:K.repeat_elements(K.expand_dims(x, -1), extract_width, -1))(address)
		patch_slices = Lambda(lambda x: x[:,:,slice_begin::])(patch)
		assert_shape(patch_slices, [self.patch_size, extract_width])

		rows = multiply([patch_slices, address_repeated])
		row = Lambda(lambda x: K.sum(x,-2))(rows)
		assert_shape(row, [extract_width])

		return row 

	def resolve_address(self, address, patch):
		assert_shape(address, [self.patch_size])
		assert_shape(patch, [self.patch_size, self.patch_width])
		return self.patch_extract(address, patch, -self.memory_size) 

	def read(self, memory, address):
		address_repeated = Lambda(lambda x:K.repeat_elements(K.expand_dims(x, -1), self.word_size, -1))(address)
		read_rows = multiply([memory, address_repeated])
		read = Lambda(lambda x: K.sum(x,-2))(read_rows)

		assert_shape(read, [self.word_size])

		return read

	def write(self, memory, address, write):
		assert_shape(memory, self.memory_shape)
		assert_shape(write, [self.word_size])
		assert_shape(address, [self.memory_size])

		address_expanded = expand_dims(address, -1)
		write = expand_dims(write, 1)
		write_e = dot([address_expanded, write], axes=[2,1], name="WriteExpanded")
		memory = add([memory, write_e], name="MemoryWrite")
		return memory

	def erase(self, memory, address, erase):
		assert_shape(memory, self.memory_shape)
		assert_shape(erase, [self.word_size])
		assert_shape(address, [self.memory_size])

		erase = expand_dims(erase, 1)
		address_expanded = expand_dims(address, -1)
		erase_e = dot([address_expanded, erase], axes=[2,1], name="EraseExpanded")
		assert_shape(erase_e, self.memory_shape)
		erase_mask = Lambda(lambda x: 1.0 - x)(erase_e)
		memory = multiply([memory, erase_mask])
		return memory

	def generate_address(self, input_data, patch, name):
		address_ptr = Dense(self.patch_size, activation="softplus",name=name)(input_data)
		address = self.resolve_address(address_ptr, patch)
		return address


class PatchSimple(PatchBase):

	def __init__(self, experiment):
		PatchBase.__init__(self, experiment)

	def build(self):

		patch = Input((self.patch_size, self.patch_width), name="InputPatch")
		memory_tm1 = Input((self.memory_shape), name="Memory")

		memory_t = memory_tm1

		# v = self.combine_nodes(patch, 2)

		flat_patch = Reshape([self.patch_size*self.patch_width])(patch)
		# first_node = Lambda(lambda x: x[:self.patch_width])(flat_patch)

		
		# ------- Memory operations --------- #
		erase_word = Dense(self.word_size, name="DenseEraseWord")(flat_patch)
		address = self.generate_address(flat_patch, patch, name="address_erase")
		memory_t = self.erase(memory_t, address, erase_word)
	
		write_word = Dense(self.word_size, name="DenseWriteWord")(flat_patch)
		address = self.generate_address(flat_patch, patch, name="address_write")
		memory_t = self.write(memory_t, address, write_word)

		# Read after so it can loopback in a single step if it wants
		address = self.generate_address(flat_patch, patch, name="address_read")
		read = self.read(memory_t, address)
		
		# out = Concatenate()([v, read])
		out = Dense(1)(read)

		return RecurrentModel(
			input=patch,
			output=out,
			return_sequences=True,
			# stateful=True,

			initial_states=[memory_tm1],
			final_states=[memory_t],
			state_initializer=[initializers.random_normal(stddev=1.0)]
		)


