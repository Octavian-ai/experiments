

import keras.backend as K
from keras.utils.test_utils import keras_test
from keras.models import Model
from keras.layers import *

from recurrentshop import RecurrentModel

import numpy as np
from numpy.testing import *

import random
from collections import namedtuple
from tensorflow import float32
from unittest import TestCase

from graph_ml import Train, Dataset
from graph_ml import PatchBase
from experiment import Experiment, ExperimentHeader

Args = namedtuple('DummyArgs', 'batch_size')


class Tests(TestCase):

	@keras_test
	def test_memory_ops(self):
	    
		memory_size = 10
		word_size = 4
		batch_size = 1

		header = ExperimentHeader(params={"word_size":word_size, "memory_size":memory_size, "patch_size":4, "patch_width":4})
		experiment = Experiment("test_memory_cell", header, Args(batch_size))

		# Initialise memory with zeros
		memory_tm1 = K.constant(np.zeros((batch_size, memory_size, word_size)), name="memory",dtype=float32)
		memory_t = memory_tm1

		# Write address is random int
		address_d = random.randint(0,memory_size - 1)
		address_one_hot_d = np.zeros([batch_size, memory_size])
		address_one_hot_d[0][address_d] = 1.0
		address = K.constant(address_one_hot_d, name="address",dtype=float32)

		# Write random pattern
		write_d = np.random.random([batch_size, word_size])
		write = K.constant(write_d, name="write")

		pb = PatchBase(experiment)
		memory_t = pb.write(memory_t, address, write)
		read = pb.read(memory_t, address)

		memory_after_write = K.eval(memory_t)
		read_final = K.eval(read)

		erase = K.constant(np.ones([batch_size, word_size]),name="erase")
		memory_t = pb.erase(memory_t, address, erase)

		memory_after_erase = K.eval(memory_t)

		for i in range(batch_size):
			for j in range(memory_size):
				if j == address_d:
					assert_allclose(memory_after_write[i][j],write_d[0])
				else:
					assert_almost_equal(memory_after_write[i][j], 0)

		assert_almost_equal(memory_after_erase, 0)
		assert_allclose(read_final, write_d)


	@keras_test
	def test_address_resolution(self):

		# Data setup
		memory_size = 20
		word_size = 4
		batch_size = 1
		patch_size = 10
		patch_width = memory_size + 5

		header = ExperimentHeader(params={"word_size":word_size, "memory_size":memory_size, "patch_size":patch_size, "patch_width":patch_width})
		experiment = Experiment("test_memory_cell", header, Args(batch_size))

		pointer = random.randint(0,patch_size - 1)
		pointer_one_hot = np.zeros([batch_size, patch_size])
		pointer_one_hot[0][pointer] = 1.0

		patch = np.random.random([batch_size, patch_size, patch_width])

		t_patch = K.constant(patch, dtype=float32, name="patch")
		t_pointer_one_hot = K.constant(pointer_one_hot, dtype=float32, name="pointer_one_hot")
		pb = PatchBase(experiment)
		resolved = K.eval(pb.resolve_address(t_pointer_one_hot, t_patch))

		for i in range(batch_size):
			assert_almost_equal(resolved[i], patch[i][pointer][-memory_size::])



	@keras_test
	def test_address_resolution_gradient(self):

		# Data setup
		memory_size = 20
		word_size = 4
		batch_size = 1
		patch_size = 10
		patch_width = memory_size + 5

		header = ExperimentHeader(params={"word_size":word_size, "memory_size":memory_size, "patch_size":patch_size, "patch_width":patch_width})
		experiment = Experiment("test_memory_cell", header, Args(batch_size))

		pb = PatchBase(experiment)

		ptr = Input((patch_size,), name="ptr")
		patch = Input((patch_size,patch_width), name="patch")
		memory = Input((memory_size, word_size), name="memory")

		resolved = pb.resolve_address(ptr, patch)
		read = pb.read(memory, resolved)

		out = Dense(3)(read)

		model = Model([ptr, patch, memory], out)
		model.compile(loss='mse', optimizer='sgd')

		model.fit({
			"ptr": np.random.random((batch_size, patch_size)), 
			"patch": np.random.random((batch_size, patch_size, patch_width)),
			"memory": np.random.random((batch_size, memory_size, word_size)),
		}, np.random.random((batch_size, 3)))


		model.predict({
			"ptr": np.zeros((batch_size, patch_size)), 
			"patch": np.zeros((batch_size, patch_size, patch_width)),
			"memory": np.zeros((batch_size, memory_size, word_size)),
		})


	@keras_test
	def test_memory_gradient(self):

		# Data setup
		memory_size = 20
		word_size = 4
		batch_size = 1
		patch_size = 10
		patch_width = memory_size + 5

		header = ExperimentHeader(params={"word_size":word_size, "memory_size":memory_size, "patch_size":patch_size, "patch_width":patch_width})
		experiment = Experiment("test_memory_cell", header, Args(batch_size))

		pb = PatchBase(experiment)

		patch = Input((patch_size, patch_width), name="patch")
		memory_tm1 = Input((memory_size, word_size), name="memory")
		memory_t = memory_tm1

		flat_patch = Reshape((patch_size*patch_width,))(patch)

		write_word = Dense(word_size)(flat_patch)
		erase_word = Dense(word_size)(flat_patch)

		ptr = Dense(patch_size)(flat_patch)
		address = pb.resolve_address(ptr, patch)
		memory_t = pb.erase(memory_t, address, erase_word)

		ptr = Dense(patch_size)(flat_patch)
		address = pb.resolve_address(ptr, patch)
		memory_t = pb.write(memory_t, address, write_word)

		ptr = Dense(patch_size)(flat_patch)
		address = pb.resolve_address(ptr, patch)
		read = pb.read(memory_t, address)

		out = Dense(3)(read)

		model = Model([patch, memory_tm1], out)
		model.compile(loss='mse', optimizer='sgd')

		model.fit({
			"patch": np.random.random((batch_size, patch_size, patch_width)),
			"memory": np.random.random((batch_size, memory_size, word_size)),
		}, np.random.random((batch_size, 3)))


		model.predict({
			"patch": np.zeros((batch_size, patch_size, patch_width)),
			"memory": np.zeros((batch_size, memory_size, word_size)),
		})




	@keras_test
	def test_memory_rnn_gradient(self):

		# Data setup
		memory_size = 20
		word_size = 4
		batch_size = 1
		patch_size = 10
		patch_width = memory_size + 5
		sequence_length = 10

		header = ExperimentHeader(params={"word_size":word_size, "memory_size":memory_size, "patch_size":patch_size, "patch_width":patch_width})
		experiment = Experiment("test_memory_cell", header, Args(batch_size))

		pb = PatchBase(experiment)

		patch = Input((patch_size, patch_width), name="patch")
		memory_tm1 = Input((memory_size, word_size), name="memory")
		memory_t = memory_tm1

		flat_patch = Reshape((patch_size*patch_width,))(patch)

		write_word = Dense(word_size)(flat_patch)
		erase_word = Dense(word_size)(flat_patch)

		ptr = Dense(patch_size)(flat_patch)
		address = pb.resolve_address(ptr, patch)
		memory_t = pb.erase(memory_t, address, erase_word)

		ptr = Dense(patch_size)(flat_patch)
		address = pb.resolve_address(ptr, patch)
		memory_t = pb.write(memory_t, address, write_word)

		ptr = Dense(patch_size)(flat_patch)
		address = pb.resolve_address(ptr, patch)
		read = pb.read(memory_t, address)

		out = Dense(3)(read)

		rnn = RecurrentModel(input=patch, output=out, initial_states=[memory_tm1], final_states=[memory_t])
		a = Input((sequence_length, patch_size, patch_width), name="patch_seq")
		b = rnn(a)
		model = Model(a, b)
		model.compile(loss='mse', optimizer='sgd')

		model.fit({
			"patch_seq": np.random.random((batch_size, sequence_length, patch_size, patch_width)),
			# "memory": np.random.random((batch_size, memory_size, word_size)),
		}, np.random.random((batch_size, 3)))


		model.predict({
			"patch_seq": np.zeros((batch_size, sequence_length, patch_size, patch_width)),
			# "memory": np.zeros((batch_size, memory_size, word_size)),
		})













