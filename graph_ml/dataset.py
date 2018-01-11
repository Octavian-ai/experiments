
from collections import Counter, namedtuple
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
# from experiment import Experiment
from .util import *

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
		transform:Callable[[Generator[neo4j.v1.Record, None, None]], Generator[Point, None, None]] = None,
		query:Callable[[], Generator[neo4j.v1.Record, None, None]] = None,
		split:Callable[[neo4j.v1.Record], Point] = None,
		finalize_x = None):

		self.transform = transform
		self.query = query

		# TODO: migrate older experiments
		if transform is None:
			def legacy_transform(rows):
				for i in rows:
					p = split(i)
					p.x = finalize_x(p.x) if finalize_x else p.x
					yield p
			self.transform = legacy_transform

		if query is None:
			def default_query(client, cypher_query, query_params):
				return client.execute_cypher(cypher_query, query_params)

			self.query = default_query



class Dataset(object):


	# Applies a per-experiment recipe to Neo4j to get a dataset to train on
	# This performs all transformations in-memory - it is not very efficient
	@classmethod
	def get(cls, experiment):
		
		# TODO: delete this
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
			)
		}

		try:
			recipe = recipes[experiment.name]
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

		query_params = QueryParams(
			golden=experiment.params.golden, 
			dataset_name=experiment.header.dataset_name, 
			experiment=experiment.name)

		query_params.update(QueryParams(**experiment.header.params))

		data_path_params = {i:query_params[i] for i in experiment.header.lazy_params}
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
		data = list(recipe.transform(data))
		total_data = len(data)
		logger.info(f"Number of rows of data: {total_data}")

		def generate_partitions():
			random.shuffle(data)
			while True:
				c = 0
				for i in data:
					
					if c == 9:
						l = "test"
					elif c == 8:
						l = "validate"
					else:
						l = "train"

					c = (c + 1) % 10

					yield (l, i)

		stream = generate_partitions()

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
		self.test_generator 		= chunk(just("test"), bs)

		# logger.info(f"First training item: {self.train_generator.peek()}")

		# These are not exact counts since the data is randomly split at generation time
		self.validation_steps 	= math.ceil(total_data * 0.1 / experiment.params.batch_size)
		self.test_steps 		= math.ceil(total_data * 0.1 / experiment.params.batch_size)
		self.steps_per_epoch 	= math.ceil(total_data * 0.8 / experiment.params.batch_size) * int(experiment.header.params.get('repeat_batch', 1))

		self.input_shape = self.train_generator.peek()[0][0].shape


class DatasetHelpers(object):

	@staticmethod
	def ensure_length(arr, length):
		delta = length - arr.shape[0]
		if delta > 0:
			pad_shape = ((0,delta),)
			for i in range(len(arr.shape)-1):
				pad_shape += ((0, 0),)
			arr = np.pad(arr, pad_shape, 'constant', constant_values=0.0)
		elif delta < 0:
			arr = arr[:length]

		assert(len(arr) == length, f"ensure_length failed to resize, {len(arr)} != {length}")

		return arr

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
	def review_from_all_hidden_random_walks(experiment):

		encode_label = {
			"NODE":	   [1,0,0,0,0],
			"PERSON":  [0,1,0,0,0],
			"REVIEW":  [0,0,1,0,0],
			"PRODUCT": [0,0,0,1,0],
			"LOOP":	   [0,0,0,0,1]
		}

		FakeNode = namedtuple('FakeNode', ['id', 'properties', 'labels'])
		loop_node = FakeNode(None, {}, set(['NODE', 'LOOP']))

		def extract_label(l):
			return encode_label.get(list(set(l) - set('NODE'))[0], [1,0,0,0])

		node_id_dict = {}

		def node_id_to_memory_addr(nid):

			if nid not in node_id_dict:
				node_id_dict[nid] = len(node_id_dict) % experiment.header.params['memory_size']

			return node_id_dict[nid]

		def package_node(n, is_target=False):
			ms = experiment.header.params['memory_size']

			if experiment.header.params["generate_address"]:
				address_trunc = node_id_to_memory_addr(n.id)
				address_one_hot = np.zeros(ms)
				address_one_hot[address_trunc] = 1.0
			else:
				address_one_hot = np.array([])

			label = extract_label(n.labels)
			score = n.properties.get("score", -1.0)

			if random.random() < experiment.header.params["target_dropout"] or is_target:
				score = -1.0

			x = np.concatenate(([score, float(is_target)], label, address_one_hot))

			return x


		def path_to_patch(node, path):
			ps = np.array([package_node(i, i.id == node.id) for i in path.nodes])

			if path.nodes[0].id == path.nodes[-1].id:
				print("outputting loop_node for ", path.nodes[0].id, [i.id for i in path.nodes])
				l = np.array([package_node(loop_node, False)])
				np.append(ps, l, axis=0)

			ps = np.repeat(ps, 2, axis=0)

			patch_size = experiment.header.params["patch_size"]
			ps = DatasetHelpers.ensure_length(ps, patch_size)
			return ps


		def row_to_point(row):
			patch_size = experiment.header.params["patch_size"]
			seq_size = experiment.header.params["sequence_size"]

			neighbors = row["neighbors"]
			review = row["review"]

			x = np.array([path_to_patch(review, path) for path in neighbors])
			x = DatasetHelpers.ensure_length(x, seq_size)
			# x = np.repeat(x, 3, axis=0)

			y = row["review"].properties.get("score", -1.0)
			# y = np.repeat([y], seq_size)
			# y = np.expand_dims(y, axis=-1)

			target_shape = (seq_size, patch_size, experiment.header.params["patch_width"])
			assert x.shape == target_shape, f"{x.shape} != {target_shape}"

			return Point(x, y)

		def query(client, cypher_query, query_params):
			return client.execute_cypher_once_per_id(
				cypher_query,
				query_params,
				dataset_name=experiment.header.dataset_name,
				id_limit=experiment.header.params["id_limit"],
				id_type="REVIEW"
			)

		def balance_classes(stream):
			# ugh arch pain
			# instead pass in an arg that is a callable stream generator

			classes = [0.0, 1.0]
			last = [None, None]

			# Over-sample
			# This is imperfectly balanced as it cold-starts without last values
			for i in stream:
				for index, c in enumerate(classes):
					if np.array([i.y]).flatten()[0] == c:
						last[index] = i
						yield i
					elif last[index] is not None:
						yield last[index]
			

		def transform(stream):
			# y_count = Counter()
			# y_count[str(y)] += 1
			# print(f"Counter of y values: {[(i, y_count[i] / len(list(y_count.elements())) * 100.0) for i in y_count]}")
			stream = (row_to_point(row) for row in stream)
			stream = balance_classes(stream)
			return stream

		return Recipe(transform=transform,query=query)

	@staticmethod
	def review_from_all_hidden_adj(experiment) -> Recipe:

		def transform(stream):
			data = list(stream)

			person_product = {}

			products = set()
			people = set()

			# Construct adjacency dict
			for i in data:
				if i["person_id"] not in person_product:
					person_product[i["person_id"]] = {}

				person_product[i["person_id"]][i["product_id"]] = i["score"]

				products.add(i["product_id"])
				people.add(i["person_id"])

			def exists(person, product):
				return 1.0 if person in person_product and product in person_product[person] else 0.0

			def score(person, product):
				return person_product.get(person, -1).get(product, -1) 

			pr_c = experiment.header.params["product_count"]
			pe_c = experiment.header.params["person_count"]

			def build(fn):
				return DatasetHelpers.ensure_length(np.array([
					DatasetHelpers.ensure_length(
						np.array([score(i, j) for j in products])
					, pr_c) for i in people
				]), pe_c)

			adj_score = build(score)
			adj_con = build(exists)

			assert_mtx_shape(adj_score, (pe_c, pr_c), "adj_score")
			assert_mtx_shape(adj_con, (pe_c, pr_c))

			print(adj_score)

			yield Point(adj_con, adj_score)


		return Recipe(transform=transform)










