
import collections
import random
import pickle
import os.path
import hashlib
import neo4j
from typing import Callable
import logging
import itertools
from itertools import cycle
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


	# Applies a per-experiment recipe to Neo4j to get a dataset to train on
	# This performs all transformations in-memory - it is not very efficient
	@classmethod
	def get(cls, experiment):
		
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
				DatasetHelpers.review_from_all_hidden(experiment)
			),
			'review_from_all_hidden_patch_rnn': Recipe(
				DatasetHelpers.review_from_all_hidden_patch_rnn(experiment)
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

		dataset_file = generate_output_path(experiment, '.pkl')

		if os.path.isfile(dataset_file) and experiment.params.lazy:
			logging.info(f"Opening dataset pickle {dataset_file}")
			data = pickle.load(open(dataset_file, "rb"))

		else:
			logging.info("Querying data from database")
			with SimpleNodeClient() as client:
				data = client.run(CypherQuery(experiment.header.cypher_query), global_params).data()
			pickle.dump(data, open(dataset_file, "wb"))

		# Because we run through the data every epoch, and for test and validate and train
		# and because we need to know the steps_per_epoch for Keras
		# the data is in memory for now - sorry brah

		# But all this downstream is legit generators!

		def generate_all():
			c = 0
			for i in data:

				p = recipe.split(i)
				p.x = recipe.finalize_x(p.x)
				
				if c == 9:
					l = "test"
				elif c == 8:
					l = "validate"
				else:
					l = "train"

				c = (c + 1) % 10

				yield (l, p)

		# It seems like Neo4J cannot tell us number of records to expect
		# without us fetching them ALL :(
		total_data = len(data)
		logging.info(f"Total number of datum {total_data}")

		self.stream = more_itertools.peekable(cycle(generate_all()))

		def just(tag):
			return ( (i[1].x, i[1].y) for i in self.stream if i[0] == tag)

		def chunk(it):
			chunky = more_itertools.chunked(it, experiment.params.batch_size)

			for i in chunky:
				xs = np.array([j[0] for j in i])
				ys = np.array([j[1] for j in i])
				yield (xs, ys)

		
		self.train_generator 		= chunk(just("train"))
		self.validation_generator 	= chunk(just("validate"))
		self.test_generator 		= chunk(just("test"))

		# These are not exact counts since the data is randomly split at generation time
		self.validation_steps = int(total_data * 0.1)
		self.test_steps = int(total_data * 0.1)
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
	def review_from_all_hidden(cls, experiment):
		def t(row):
			length = experiment.header.meta["neighbor_count"]
			neighbors = np.array(row["neighbors"])
			delta = length - neighbors.shape[0]

			if delta > 0:
				neighbors = np.pad(neighbors, ((0,delta), (0, 0)), 'constant', constant_values=0.0)
			
			return Point(neighbors, row["score"])

		return t

	@staticmethod
	def review_from_all_hidden_patch_rnn(experiment):

		encode_label = {
			"PERSON":  [0,1,0,0],
			"REVIEW":  [0,0,1,0],
			"PRODUCT": [0,0,0,1]
		}

		def extract_label(l):
			print(list(set(l) - set('NODE'))[0])
			return encode_label.get(list(set(l) - set('NODE'))[0], [1,0,0,0])

		def package_node(n, l, hide_score=False):
			score = n.properties.get("score", -1.0)

			if random.random() < experiment.header.meta["target_dropout"] or hide_score:
				score = -1.0

			label = extract_label(l)
			return np.concatenate(([score],label, np.zeros(experiment.header.meta["state"])))

		def path_map(i):
			return package_node(i[0], i[1])

		def t(row):
			n = DatasetHelpers.collect_neighbors(row, 'neighbors', path_map, 20)
			
			h = np.concatenate(([1],package_node(row["node"], row["labels(node)"], True)))
			x = np.concatenate(([h], n))

			return Point(x, row["node"].properties.get("score", -1.0))

		return t






	



