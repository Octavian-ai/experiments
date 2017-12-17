
import collections
import random
import pickle
import os.path
import hashlib
import neo4j
from typing import Callable
import logging

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




class Dataset(object):

	@staticmethod
	def get(experiment):
		params = experiment.params
		
		dataset_file = generate_output_path(experiment, '.pkl')

		if os.path.isfile(dataset_file) and params.lazy:
			logging.info("Opening data pickle")
			d = pickle.load(open(dataset_file, "rb"))

		else:
			logging.info("Querying data from database")
			d = Dataset.generate(experiment)
			pickle.dump(d, open(dataset_file, "wb"))

		logging.info(f"Test data sample: {str(list(zip(d.test.x, d.test.y))[:10])}")

		if len(d.test.x) == 0 or len(d.train.x) == 0:
			logging.error("Dataset too small to provide test and training data, this run will fail")

		return d

	# Applies a per-experiment recipe to Neo4j to get a dataset to train on
	# This performs all transformations in-memory - it is not very efficient
	@classmethod
	def generate(cls, experiment):
		params = experiment.params

		global_params = QueryParams(golden=params.golden, dataset_name=experiment.header.dataset_name, experiment=params.experiment)

		class Recipe:
			def __init__(self, split:Callable[[neo4j.v1.Record], Point], finalize_x=lambda x:x, params:QueryParams=global_params):
				self.split = split
				self.finalize_x = finalize_x
				self.params = params

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
			'review_from_all_hidden': Recipe(
				DatasetHelpers.review_from_all_hidden(experiment.header.meta["neighbor_count"])
			)
		}

		return Dataset.execute_recipe(experiment, recipes[params.experiment])



	@classmethod
	def execute_recipe(cls, experiment, recipe):
		params = experiment.params

		with SimpleNodeClient() as client:
			data = client.execute_cypher(CypherQuery(experiment.header.cypher_query), recipe.params)

			# Once I get my shit together,
			# 1) use iterators
			# 2) move to streaming
			# 3) move to hdf5

			data = list(data) # so we can do a few passes

			if len(data) == 0:
				raise Exception('Neo4j query returned no data, cannot train the network') 

			logging.info(f"Retrieved {len(data)} rows from Neo4j")
			logging.info(f"Data sample: {str(data[:10])}")

			xy = [recipe.split(i) for i in data]

			return Dataset(params, recipe, data, xy)
		

	# Split data into test/train set, organise it into a class
	def __init__(self, params, recipe, data, xy):
		self.params = params
		self.data = data
		self.data_xy = xy

		self.input_shape = (len(xy[0].x),)

		self.train = Point([], [])
		self.test = Point([], [])

		if params.random_seed is not None:
			random.seed(params.random_seed)

		def store_datum(i):
			r = random.random()
			if r > 0.9:
				self.test.append(i)
			else:
				self.train.append(i)

		for i in self.data_xy:
			store_datum(i)

		# Yuck. fix later.
		self.train.x = recipe.finalize_x(np.array(self.train.x))
		self.train.y = np.array(self.train.y)

		self.test.x = recipe.finalize_x(np.array(self.test.x))
		self.test.y = np.array(self.test.y)



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






	



