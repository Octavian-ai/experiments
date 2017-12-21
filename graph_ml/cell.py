
import keras
import keras.backend as K

"""
		self.kernel = self.add_weight(shape=(input_shape[-1], self.units),
									  initializer='uniform',
									  name='kernel')
		self.recurrent_kernel = self.add_weight(
			shape=(self.units, self.units),
			initializer='uniform',
			name='recurrent_kernel')

		h = K.dot(inputs, self.kernel)
		output = h + K.dot(prev_output, self.recurrent_kernel)


		cell: A RNN cell instance. A RNN cell is a class that has:
			- a `call(input_at_t, states_at_t)` method, returning
				`(output_at_t, states_at_t_plus_1)`. The call method of the
				cell can also take the optional argument `constants`, see
				section "Note on passing external constants" below.
			- a `state_size` attribute. This can be a single integer
				(single state) in which case it is
				the size of the recurrent state
				(which should be the same as the size of the cell output).
				This can also be a list/tuple of integers
				(one size per state). In this case, the first entry
				(`state_size[0]`) should be the same as
				the size of the cell output.
			It is also possible for `cell` to be a list of RNN cell instances,
			in which cases the cells get stacked on after the other in the RNN,
			implementing an efficient stacked RNN.
"""

class AddressableCell(keras.layers.Layer):

	def __init__(self, memory_size, **kwargs):
		self.memory_size = memory_size
		super(AddressableCell, self).__init__(**kwargs)

	def build(self, input_shape):
		self.built = True


	# TODO: work out actual fns
	def read(self, memory, address):
		return K.dot(memory, address)

	def write(self, memory, address, write):
		return memory + K.dot(address, write)

	def erase(self, memory, address, erase):
		return K.dot(memory, (1 - K.dot(address, erase)))
	# -----------------------------

	def call(self, inputs, states):
		address, write, erase = inputs

		memory = states[0]
		output = self.read(memory, address)
		memory = self.erase(memory, address, erase)
		memory = self.write(memory, address, write)

		return output, [memory]

class PatchCell(keras.layers.Layer):

	


