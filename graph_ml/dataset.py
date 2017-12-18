
import collections
import random
import pickle
import os.path
import hashlib
import neo4j
from typing import Callable
import logging
import itertools
import more_itertools

import keras
import numpy as np
from keras.preprocessing import text
from keras.utils import np_utils

import experiment
from .path import generate_output_path
from graph_io import *



class Point(object):
	def __init__(self, x, y):
		self.x = x
		self.y = y

	# This is weird, I know, re-write later when I'm making this more efficient
	def append(self, point):
		self.x.append(point.x)
		self.y.append(point.y)

	def __str__(self):
		return "{x: " + str(self.x) + ",\ny: " + str(self.y) + "}"


class Recipe:
	def __init__(self, split:Callable[[neo4j.v1.Record], Point], finalize_x=lambda x:x):
		self.split = split
		self.finalize_x = finalize_x


class Dataset(object):

	@staticmethod
	def get(experiment):
		params = experiment.params
		
		dataset_file = generate_output_path(experiment, '.pkl')
		david_has_made_this_work_for_generators = False

		if os.path.isfile(dataset_file) and params.lazy and david_has_made_this_work_for_generators:
			logging.info(f"Opening dataset pickle {dataset_file}")
			d = pickle.load(open(dataset_file, "rb"))

		else:
			logging.info("Querying data from database")
			d = Dataset.generate(experiment)

			if david_has_made_this_work_for_generators:
				pickle.dump(d, open(dataset_file, "wb"))
				logging.info(f"Saved dataset pickle {dataset_file}")

		# logging.info(f"Test data sample: {str(list(zip(d.test.x, d.test.y))[:1])}")

		# if len(d.test.x) == 0 or len(d.train.x) == 0:
		# 	logging.error("Dataset too small to provide test and training data, this run will fail")

		return d

	# Applies a per-experiment recipe to Neo4j to get a dataset to train on
	# This performs all transformations in-memory - it is not very efficient
	@classmethod
	def generate(cls, experiment):
		
		recipes = {
			'review_from_visible_style': Recipe(
				lambda row: Point(np.concatenate((row['style_preference'], row['style'])), row['score'])
			),
			'review_from_hidden_style_neighbor_conv': Recipe(
				DatasetHelpers.review_from_hidden_style_neighbor_conv(100),
				lambda x: {'person':np.array([i['person'] for i in x]), 'neighbors': np.array([i['neighbors'] for i in x])}
			),
			'style_from_neighbor_conv': Recipe(
				DatasetHelpers.style_from_neighbor(100)
			),
			'style_from_neighbor_rnn': Recipe(
				DatasetHelpers.style_from_neighbor(100)
			),
			'review_from_all_hidden_simple_unroll': Recipe(
				DatasetHelpers.review_from_all_hidden(experiment.header.meta["neighbor_count"])
			),
			'review_from_all_hidden_patch_rnn': Recipe(
				DatasetHelpers.review_from_all_hidden_patch_rnn
			)
		}

		return Dataset(experiment, recipes[experiment.name])

		

	# Split data into test/train set, organise it into a class
	def __init__(self, experiment, recipe):

		self.experiment = experiment
		self.recipe = recipe

		if experiment.params.random_seed is not None:
			random.seed(params.random_seed)

		global_params = QueryParams(
			golden=experiment.params.golden, 
			dataset_name=experiment.header.dataset_name, 
			experiment=experiment.name)

		with SimpleNodeClient() as client:

			def run_query():
				return client.run(CypherQuery(experiment.header.cypher_query), global_params)

			self.statement_result = run_query()

			def generate_all():
				for i in self.statement_result:

					p = recipe.split(i)
					p.x = recipe.finalize_x(p.x)

					r = random.random()
					if r > 0.9:
						l = "test"
					elif r > 0.8:
						l = "validate"
					else:
						l = "train"

					yield (l, p)

			# It seems like Neo4J cannot tell us number of records to expect
			# without us fetching them ALL :(
			total_data = len(run_query().data())

			def chunk_train_data():
				just_trainy = (i[1] for i in self.stream if i[0] == "train")
				chunky = more_itertools.chunked(just_trainy, experiment.params.batch_size)

				for i in chunky:
					xs = np.array([j.x for j in i])
					ys = np.array([j.y for j in i])
					yield (xs, ys)

			self.stream = more_itertools.peekable(generate_all())
			self.train_generator 		= itertools.cycle(chunk_train_data())
			self.validation_generator 	= itertools.cycle(((i[1].x, i[1].y) for i in self.stream if i[0] == "validate"))
			self.test_generator 		= ((i[1].x, i[1].y) for i in self.stream if i[0] == "test")
			
			# These are not exact counts since the data is randomly split at generation time
			self.validation_steps = int(total_data * 0.1)
			self.steps_per_epoch = int(total_data * 0.8 / experiment.params.batch_size)

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
	def review_from_all_hidden(cls, length):
		def t(row):
			neighbors = np.array(row["neighbors"])
			delta = length - neighbors.shape[0]

			if delta > 0:
				neighbors = np.pad(neighbors, ((0,delta), (0, 0)), 'constant', constant_values=0.0)
			
			return Point(neighbors, row["score"])

		return t

	@staticmethod
	def review_from_all_hidden_patch_rnn(experiment):

		def extract_label(l):
			return list(l - set('NODE'))[0]

		def package_node(n,l):
			score = n.properties.get("score", -1.0)

			if random.random() < experiment.header.meta["target_dropout"]:
				score = -1.0

			label = extract_label(l)
			return [score,label, np.zero(experiment.header.meta["state"])]

		def path_map(i):
			return package_node(i[0], i[1])

		def t(row):
			n = DatasetHelpers.collect_neighbors(row, 'neighbors', path_map, 20)
			x = (package_node(row["node"], row["labels(node)"]), n)

			return Point(x, row["node"].properties["score"])

		return t






	



