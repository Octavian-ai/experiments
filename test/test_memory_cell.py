

import keras.backend as K
from keras.utils.test_utils import keras_test
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
	def test_memory_cell(self):
	    
		memory_size = 10
		word_size = 4
		batch_size = 1

		header = ExperimentHeader(params={"word_size":word_size, "memory_size":memory_size, "patch_size":4, "patch_width":4})
		experiment = Experiment("test_memory_cell", header, Args(batch_size))

		# Initialise memory with zeros
		memory_tm1 = K.constant(np.zeros((batch_size, memory_size, word_size)), name="memory",dtype=float32)

		# Write address is random int
		address_d = random.randint(0,memory_size-1)
		address_one_hot_d = np.zeros([batch_size, memory_size])
		address_one_hot_d[0][address_d] = 1.0
		address = K.constant(address_one_hot_d, name="address",dtype=float32)

		# Write random pattern
		write_d = np.random.random([batch_size, word_size])
		write = K.constant(write_d, name="write")

		pb = PatchBase(experiment)
		memory_t = pb.write(memory_tm1, address, write)
		read = pb.read(memory_t, address)

		memory_final = K.eval(memory_t)
		read_final = K.eval(read)

		for i in range(batch_size):
			for j in range(memory_size):
				if j == address_d:
					assert_allclose(memory_final[i][j],write_d[0])
				else:
					assert_almost_equal(memory_final[i][j], 0)


		assert_allclose(read_final, write_d)


