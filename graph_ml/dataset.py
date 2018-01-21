
from collections import Counter, namedtuple
import random
import pickle
import os.path
import hashlib
import neo4j
import math
from typing import Callable, Generator, Tuple
import logging
import itertools
from itertools import cycle
import more_itertools
from more_itertools import peekable

import keras
import numpy as np
from keras.preprocessing import text
from keras.utils import np_utils

from .path import generate_output_path, generate_data_path
from graph_io import *
# from experiment import Experiment
from .util import *
from .dataset_helpers import *

logger = logging.getLogger(__name__)


class Dataset(object):


	# Applies a per-experiment recipe to Neo4j to get a dataset to train on
	# This performs all transformations in-memory - it is not very efficient
	@classmethod
	def get(cls, experiment):
		
		# TODO: delete this
		legacy_recipes = {
			'review_from_visible_style': Recipe(
				split=lambda row: Point(np.concatenate((row['style_preference'], row['style'])), row['score'])
			),
			'review_from_hidden_style_neighbor_conv': Recipe(
				split=DatasetHelpers.review_from_hidden_style_neighbor_conv(100),
				finalize_x=lambda x: {'person':np.array([i['person'] for i in x]), 'neighbors': np.array([i['neighbors'] for i in x])}
			),
			'style_from_neighbor_conv': Recipe(
				split=DatasetHelpers.style_from_neighbor(100)
			),
			'style_from_neighbor_rnn': Recipe(
				split=DatasetHelpers.style_from_neighbor(100)
			)
		}

		try:
			recipe = legacy_recipes[experiment.name]
		except:
			# TODO: move all to this pattern
			recipe = getattr(DatasetHelpers, experiment.name)(experiment)


		return Dataset(experiment, recipe)

		

	# Split data into test/train set, organise it into a class
	def __init__(self, experiment, recipe):

		self.experiment = experiment
		self.recipe = recipe

		if experiment.params.random_seed is not None:
			random.seed(experiment.params.random_seed)

		if experiment.params.dataset_name is not None:
			dataset_name = experiment.params.dataset_name
		else:
			dataset_name = experiment.header.dataset_name

		query_params = QueryParams(
			golden=experiment.params.golden, 
			dataset_name=dataset_name, 
			experiment=experiment.name)

		query_params.update(QueryParams(**experiment.header.params))

		# Calculate params for lazy data loading
		data_path_params = {i:query_params[i] for i in experiment.header.lazy_params}
		data_path_params["dataset_name"] = dataset_name

		dataset_file = generate_data_path(experiment, '.pkl', data_path_params)
		logger.info(f"Dataset file {dataset_file}")

		if os.path.isfile(dataset_file) and experiment.params.lazy:
			logger.info(f"Opening dataset pickle {dataset_file}")
			data = pickle.load(open(dataset_file, "rb"))

		else:
			logger.info("Querying data from database")
			with SimpleNodeClient() as client:
				cq = CypherQuery(experiment.header.cypher_query)
				data = recipe.query(client, cq, query_params)

				# Later shift to query-on-demand
				data = list(data)
			pickle.dump(data, open(dataset_file, "wb"))

		# We need to know total length of data, so for ease I've listed it here.
		# I've used generators everywhere, so if it wasn't for Keras, this would
		# be memory efficient
		
		logger.info(f"Rows returned by Neo4j {len(data)}")
		list_data = list(recipe.transform(data))
		total_data = len(list_data)
		logger.info(f"Number of rows of data: {total_data}")


		def repeat_infinitely(gen_fn):
			while True:
				for x in gen_fn():
					yield x
		stream = repeat_infinitely(lambda: recipe.partition(recipe.transform(data)))

		def just(tag):
			return ( (i[1].x, i[1].y) for i in stream if i[0] == tag)

		def chunk(it, length):
			chunky = more_itertools.chunked(it, length)
			for i in chunky:
				xs = np.array([j[0] for j in i])
				ys = np.array([j[1] for j in i])
				yield (xs, ys)

		
		bs = experiment.params.batch_size

		self.train_generator 		= peekable(chunk(just("train"), bs))
		self.validation_generator 	= peekable(chunk(just("validate"), bs))
		self.test_generator 		= peekable(chunk(just("test"), bs))

		self.generator = {
			"test": self.test_generator,
			"train": self.train_generator,
			"validate": self.validation_generator
		}

		f = self.train_generator.peek()
		# logger.info(f"First training item: x:{f[0].shape}, y:{f[1].shape}")

		# These are not exact counts since the data is randomly split at generation time
		self.validation_steps 	= math.ceil(total_data * 0.1 / experiment.params.batch_size)
		self.test_steps 		= math.ceil(total_data * 0.1 / experiment.params.batch_size)
		self.steps_per_epoch 	= math.ceil(total_data * 0.8 / experiment.params.batch_size) * int(experiment.header.params.get('repeat_batch', 1))

		self.input_shape = self.train_generator.peek()[0][0].shape





