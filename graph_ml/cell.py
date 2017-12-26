
import keras
import keras.backend as K

from keras.models import Model
from keras.layers import *
from recurrentshop import RecurrentModel

from .util import *

class PatchBase(object):

	def __init__(self, experiment):
		self.experiment = experiment

		self.patch_size  = experiment.header.meta["patch_size"]
		self.patch_width = experiment.header.meta["patch_width"]

		self.word_size = self.experiment.header.meta["word_size"]
		self.batch_size = self.experiment.params.batch_size
		self.memory_size = self.experiment.header.meta["memory_size"]

		self.word_shape = [self.word_size]
		self.word_shape_batch = [self.batch_size, self.word_size]
		self.memory_shape = [self.memory_size, self.word_size]
		self.memory_shape_batch = [None] + self.memory_shape

	def combine_nodes(self, patch, width):
		n = Conv1D(
			filters=width, 
			kernel_size=1, 
			activation='tanh', 
			kernel_initializer='random_uniform',
			bias_initializer='zeros',
			name="ConvPatch")(patch)

		n = MaxPooling1D(self.patch_size)(n)
		n = Reshape([width])(n)
		return n

	def resolve_address(self, address, patch):
		address_repeated = Lambda(lambda x:K.repeat_elements(K.expand_dims(x, -1), self.memory_size, -1))(address)
		patch_addresses = Lambda(lambda x: x[:,:,-self.memory_size::],name="PatchAddressBlock")(patch)
		assert_shape(patch_addresses, (self.patch_size, self.memory_size))

		address_rows = multiply([patch_addresses, address_repeated])
		address_resolved = Lambda(lambda x: K.sum(x,-2))(address_rows)
		assert_shape(address_resolved, [self.memory_size])

		return address_resolved 

	def read(self, memory, address):
		address_repeated = Lambda(lambda x:K.repeat_elements(K.expand_dims(x, -1), self.word_size, -1))(address)
		read_rows = multiply([memory, address_repeated])
		read = Lambda(lambda x: K.sum(x,-1))(read_rows)
		return read

	def write(self, memory, address, write):
		address_expanded = expand_dims(address, -1)
		write = expand_dims(write, 1)
		write_e = dot([address_expanded, write], axes=[2,1], name="WriteExpanded")
		memory = add([memory, write_e], name="MemoryWrite")
		return memory

	def erase(self, memory, address, erase):
		erase = expand_dims(erase, 1)
		address_expanded = expand_dims(address, -1)
		erase_e = dot([address_expanded, erase], axes=[2,1], name="EraseExpanded")
		assert_shape(erase_e, self.memory_shape)
		erase_mask = Lambda(lambda x: 1.0 - x)(erase_e)
		memory = multiply([memory, erase_mask])
		return memory


class PatchSimple(PatchBase):

	def __init__(self, experiment):
		PatchBase.__init__(self, experiment)

	

	def build(self):

		patch = Input([self.patch_size, self.patch_width], name="InputPatch")

		memory_tm1 = Input(self.memory_shape, name="Memory")
		memory_t = memory_tm1

		v = self.combine_nodes(patch, 5)
		address_ptr = Dense(self.patch_size)(v)
		address = self.resolve_address(address_ptr, patch)
		
		# Memory operations
		write_word = Dense(self.word_size, name="DenseWriteWord")(v)
		memory_t = self.write(memory_t, address, write_word)

		erase_word = Dense(self.word_size, name="DenseEraseWord")(v)
		memory_t = self.erase(memory_t, address, erase_word)

		# Read after so it can loopback in a single step if it wants
		read = self.read(memory_t, address)
		
		# v = Concatenate()([v, read])
		out = Dense(5)(read)

		return RecurrentModel(
			input=patch,
			output=out,
			return_sequences=True,

			initial_states=[memory_tm1],
			final_states=[memory_t],
			state_initializer=[initializers.random_normal(stddev=1.0)]
		)



class PatchRNN(PatchBase):

	def __init__(self, experiment):
		PatchBase.__init__(self, experiment)


	def __call__(self, patch_in):
		batch_size  = self.experiment.params.batch_size
		patch_size  = self.experiment.header.meta["patch_size"]
		patch_width = self.experiment.header.meta["patch_width"]
		memory_size = self.experiment.header.meta["memory_size"]
		word_size   = self.experiment.header.meta["word_size"]
		node_control_width = self.experiment.header.meta["node_control_width"]

		# patch = Input(batch_shape=(batch_size, patch_size, patch_width), name="InputPatch")
		# memory_in = Input(batch_shape=(batch_size, memory_size, word_size), name="InputMemory")
		# patch = Input((patch_size, patch_width), name="InputPatch")
		memory_in = Input((memory_size, word_size), name="InputMemory")
		patch_flat = Input([patch_size*patch_width], name="InputPatch")

		patch = Reshape([patch_size, patch_width])(patch_flat)

		assert_shape(patch, (patch_size, patch_width))
		assert_shape(memory_in, (memory_size, word_size))

		all_control = self.combine_nodes(patch, node_control_width)

		address_reference = Dense(patch_size)(all_control)
		write = Dense(word_size)(all_control)
		erase = Dense(word_size)(all_control)

		address_resolved = self.resolve_address(address_reference, patch)

		assert_shape(address_resolved, [memory_size])
		assert_shape(erase, [word_size])
		assert_shape(write, [word_size])

		out, memory_out = self.memory_op(memory_in, address_resolved, write, erase)

		# out = Concatenate()([all_control, out]) # Combo
		out = Dense(patch_width,name="DenseFinalOut")(out) # Normal option

		# return Model([patch, memory_in], [out, memory_out])
		return RecurrentModel(
			input=patch_flat,
			output=out,
			initial_states=[memory_in],
			final_states=[memory_out],
			state_initializer=[initializers.zeros()],
			return_sequences=True
		)(patch_in)



# class PatchRNN(object):

# 	def __init__(self, experiment):
# 		self.experiment = experiment

# 	def __call__(self, layer_in):
# 		patch_size  = self.experiment.header.meta["patch_size"]
# 		patch_width = self.experiment.header.meta["patch_width"]
# 		cell = PatchCell(self.experiment, output_dim=patch_width, input_shape=(patch_size, patch_width))
# 		# get_layer accepts arguments like return_sequences, unroll etc :
# 		return cell.get_layer(return_sequences=True)(layer_in)




