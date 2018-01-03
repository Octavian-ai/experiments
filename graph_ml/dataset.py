
from collections import Counter
import random
import pickle
import os.path
import hashlib
import neo4j
import math
from typing import Callable, Generator
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

logger = logging.getLogger(__name__)


class Point(object):
	def __init__(self, x, y):
		self.x = x
		self.y = y

	# This is weird, I know, re-write later when I'm making this more efficient
	def append(self, point):
		self.x.append(point.x)
		self.y.append(point.y)

	def __str__(self):
		return "{x:\n" + str(self.x) + ",\ny:\n" + str(self.y) + "}"

	def __repr__(self):
		return self.__str__()


def noop():
	pass

class Recipe:
	def __init__(self, 
		transform=Generator[neo4j.v1.Record,None,Point], 
		split:Callable[[neo4j.v1.Record], Point]=None, 
		finalize_x=None):

		self.transform = transform

		# TODO: migrate older experiments
		if transform is None:
			def legacy_transform(rows):
				for i in rows:
					p = split(i)
					p.x = finalize_x(p.x)
					yield p
			self.transform = legacy_transform


class Dataset(object):


	# Applies a per-experiment recipe to Neo4j to get a dataset to train on
	# This performs all transformations in-memory - it is not very efficient
	@classmethod
	def get(cls, experiment):
		
		recipes = {
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
			),
			'review_from_all_hidden_simple_unroll': Recipe(
				split=DatasetHelpers.review_from_all_hidden(experiment)
			),
			'review_from_all_hidden_ntm': DatasetHelpers.review_from_all_hidden_ntm(experiment)
		}

		return Dataset(experiment, recipes[experiment.name])

		

	# Split data into test/train set, organise it into a class
	def __init__(self, experiment, recipe):

		self.experiment = experiment
		self.recipe = recipe

		if experiment.params.random_seed is not None:
			random.seed(params.random_seed)

		query_params = QueryParams(
			golden=experiment.params.golden, 
			dataset_name=experiment.header.dataset_name, 
			experiment=experiment.name)

		query_params.update(experiment.header.params)

		dataset_file = generate_data_path(experiment, '.pkl')
		logger.info(f"Dataset file {dataset_file}")

		if os.path.isfile(dataset_file) and experiment.params.lazy:
			logger.info(f"Opening dataset pickle {dataset_file}")
			data = pickle.load(open(dataset_file, "rb"))

		else:
			logger.info("Querying data from database")
			with SimpleNodeClient() as client:
				data = client.run(CypherQuery(experiment.header.cypher_query), query_params).data()
			pickle.dump(data, open(dataset_file, "wb"))

		# We need to know total length of data, so for ease I've listed it here.
		# I've used generators everywhere, so if it wasn't for Keras, this would
		# be memory efficient
		
		logger.info(f"Rows returned by Neo4j {len(data)}")
		data = list(recipe.transform(data))
		total_data = len(data)
		logger.info(f"Number of rows of data: {total_data}")

		def generate_partitions():
			while True:
				c = 0
				random.shuffle(data)
				for i in data:
					
					if c == 9:
						l = "test"
					elif c == 8:
						l = "validate"
					else:
						l = "train"

					c = (c + 1) % 10

					yield (l, i)

		self.stream = peekable(generate_partitions())

		def just(tag):
			return ( (i[1].x, i[1].y) for i in self.stream if i[0] == tag)

		def chunk(it, length):
			chunky = more_itertools.chunked(it, length)
			for i in chunky:
					xs = np.array([j[0] for j in i])
					ys = np.array([j[1] for j in i])
					yield (xs, ys)

		def chunk_key(it, length, keys):
			chunky = more_itertools.chunked(it, length)

			def c(k):
				for i in chunky:
					xs = np.array([j[0][k] for j in i])
					ys = np.array([j[1] for j in i])
					yield (xs, ys)

			return {k: c(k) for k in keys}

		
		bs = experiment.params.batch_size
		ss = experiment.header.params["sequence_size"]

		def chunk_chunk(it, keys=None):
			return chunk(chunk(it, ss), bs)

		def chunk_chunk_key(it, key):
			return chunk_key(chunk_key(it, ss, key), bs, key)

		keys = ["neighbor", "node"]

		self.train_generator 		= peekable(chunk_chunk(just("train"), keys))
		self.validation_generator 	= peekable(chunk_chunk(just("validate"), keys))
		self.test_generator 		= chunk_chunk(just("test"), keys)

		# logger.info(f"First training item: {self.train_generator.peek()}")

		# These are not exact counts since the data is randomly split at generation time
		self.validation_steps 	= math.ceil(total_data * 0.1 / experiment.params.batch_size)
		self.test_steps 		= math.ceil(total_data * 0.1 / experiment.params.batch_size)
		self.steps_per_epoch 	= math.ceil(total_data * 0.8 / experiment.params.batch_size) * int(experiment.header.params.get('repeat_batch', 1))

		self.input_shape = (len(self.stream.peek()[0]),)


class DatasetHelpers(object):

	@staticmethod
	def path_map_style_preference_score(cls, path):
		other_person = path.nodes[0]
		other_review = path.nodes[1]
		return np.concatenate((
				np.array(other_person.properties['style_preference']),
				[other_review.properties['score']]
			))

	# Turn neighbors sub-graph into a sampled array of neighbours
	# @argument length What size of array should be returned. Use None for variable. If you request a fixed length, the first column of the feature is a 0.0/1.0 flag of where there is data or zeros in that feature row
	@classmethod
	def collect_neighbors(cls, row, key, path_map, length:int):
		subrows = []
		for path in row[key]:
			subrows.append(path_map(path))

		# Lets always shuffle to keep the network on its toes
		# If you use --random-seed you'll fix this to be the same each run
		np.random.shuffle(subrows)

		if length is not None:
			if len(subrows) > length:
				subrows = subrows[:length]
	
			subrows = np.pad(subrows, ((0,0), (1,0)), 'constant', constant_values=1.0) # add 'none' flag

			# pad out if too small
			# note if there are zero subrows, this won't know the width to make the zeros, so it'll be 1 wide and broadcast later
			if len(subrows) < length:
				delta = length - subrows.shape[0]
				subrows = np.pad(subrows, ((0,delta), (0, 0)), 'constant', constant_values=0.0)

		return subrows


	@classmethod
	def review_from_hidden_style_neighbor_conv(cls, length):
		def transform_row(row):
			neighbors = cls.collect_neighbors(row, 'neighbors', cls.path_map_style_preference_score)
			return Point({'person': np.array(row["style_preference"]), 'neighbors':neighbors}, row["score"])
		return transform_row


	@classmethod
	def style_from_neighbor(cls, length):
		# Python you suck at developer productivity.
		# Seriously, coffeescript has all these things sorted out
		# Like no anonymous functions? Fuck you.
		def transform_row(row):
			neighbors = cls.collect_neighbors(row, 'neighbors', cls.path_map_style_preference_score, length)
			return Point(neighbors, row["product"].properties["style"])
		return transform_row


	@classmethod
	def review_from_all_hidden(cls, experiment):
		def t(row):
			length = experiment.header.params["neighbor_count"]
			neighbors = np.array(row["neighbors"])
			delta = length - neighbors.shape[0]

			if delta > 0:
				neighbors = np.pad(neighbors, ((0,delta), (0, 0)), 'constant', constant_values=0.0)
			
			return Point(neighbors, row["score"])

		return t

	@staticmethod
	def review_from_all_hidden_ntm(experiment):

		encode_label = {
			"PERSON":  [0,1,0,0],
			"REVIEW":  [0,0,1,0],
			"PRODUCT": [0,0,0,1]
		}

		def extract_label(l):
			return encode_label.get(list(set(l) - set('NODE'))[0], [1,0,0,0])

		node_id_dict = {}

		def node_id_to_memory_addr(nid):

			if nid not in node_id_dict:
				node_id_dict[nid] = len(node_id_dict) % experiment.header.params['memory_size']

			return node_id_dict[nid]

		def package_node(n, l, is_head=0.0, hide_score=False):
			ms = experiment.header.params['memory_size']

			address_trunc = node_id_to_memory_addr(n.id)
			address_one_hot = np.zeros(ms)
			address_one_hot[address_trunc] = 1.0

			label = extract_label(l)
			score = n.properties.get("score", -1.0)

			if random.random() < experiment.header.params["target_dropout"] or hide_score:
				score = -1.0

			x = np.concatenate(([is_head, score], label, address_one_hot))
			
			return x

		def row_to_point(row):
			patch_size = experiment.header.params["patch_size"]

			x = np.array([package_node(i, i.labels, (1.0 if n==0 else 0.0), n==0) for n, i in enumerate(row["g"].nodes[:patch_size])])

			# pad out if too small
			delta = patch_size - x.shape[0]
			if delta > 0:
				x = np.pad(x, ((0,delta), (0, 0)), 'constant', constant_values=0.0)

			y = row["g"].nodes[0].properties.get("score", -1.0)
			label = row["g"].nodes[0].labels

			

			target_shape = (experiment.header.params["patch_size"], experiment.header.params["patch_width"])
			assert x.shape == target_shape, f"{x.shape} != {target_shape}"

			return Point(x, [y])


		def without_dupes(stream):
			seen_nodes = set()

			for row in stream:
				nodes = set([i.id for i in row["g"].nodes])

				if seen_nodes.isdisjoint(nodes):
					seen_nodes.update(nodes)
					yield row

			# print(f"I saw nodes {seen_nodes}")


		def transform(stream):
			y_count = Counter()

			# ugh arch pain
			all_pts = list((row_to_point(row) for row in stream))

			ones  = (i for i in all_pts if i.y[0] == 1.0)
			zeros = (i for i in all_pts if i.y[0] == 0.0)

			for i in zip(ones, zeros):
				yield i[0]
				yield i[1]

			# y_count[str(y)] += 1
			# print(f"Counter of y values: {[(i, y_count[i] / len(list(y_count.elements())) * 100.0) for i in y_count]}")
			
		return Recipe(transform=transform)






	



