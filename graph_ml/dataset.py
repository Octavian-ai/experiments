
import collections
import random
import pickle
import os.path
import hashlib
import neo4j

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
	def get(params):
		dataset_file = generate_output_path(params, '.pkl')

		if os.path.isfile(dataset_file) and params.lazy:
			d = pickle.load(open(dataset_file, "rb"))

		else:
			d = Dataset.generate(params)
			pickle.dump(d, open(dataset_file, "wb"))

		if params.verbose > 0:
			print("Test data sample: ", list(zip(d.test.x, d.test.y))[:10])

		return d

	# Applies a per-experiment recipe to Neo4j to get a dataset to train on
	# This performs all transformations in-memory - it is not very efficient
	@classmethod
	def generate(cls, params):

		global_params = QueryParams(golden=params.golden, experiment=params.experiment)

		class Recipe:
			def __init__(self, split, finalize_x=lambda x:x, params=global_params):
				self.split = split
				self.finalize_x = finalize_x
				self.params = params
				

		recipes = {
			'review_from_visible_style': Recipe(
				lambda row: Point(np.concatenate((row['style_preference'], row['style'])), row['score'])
			),
			'review_from_hidden_style_neighbor_conv': Recipe(
				DatasetHelpers.review_from_hidden_style_neighbor_conv,
				lambda x: {'person':np.array([i['person'] for i in x]), 'neighbors': np.array([i['neighbors'] for i in x])}
			),
			'style_from_neighbor_conv': Recipe(
				DatasetHelpers.style_from_neighbor_conv
			)
		}

		return Dataset.execute_recipe(params, recipes[params.experiment])



	@classmethod
	def execute_recipe(cls, params, recipe):
		with SimpleNodeClient() as client:
			data = client.execute_cypher(CypherQuery(experiment.directory[params.experiment].cypher_query), recipe.params)

			# Once I get my shit together,
			# 1) use iterators
			# 2) move to streaming
			# 3) move to hdf5

			data = list(data) # so we can do a few passes

			if len(data) == 0:
				raise Exception('Neo4j query returned no data, cannot train the network') 

			if params.verbose > 1:
				print("Retrieved {} rows from Neo4j".format(len(data)))
				print("Data sample: ", data[:10])

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

	@classmethod
	def collect_neighbors(cls, row, key, length=100):
		subrows = []
		for path in row[key]:
			other_person = path.nodes[0]
			other_review = path.nodes[1]
			subrows.append(np.concatenate((
				np.array(other_person.properties['style_preference']),
				[other_review.properties['score']]
			)))

		np.random.shuffle(subrows)

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
	def review_from_hidden_style_neighbor_conv(cls, row):
		neighbors = cls.collect_neighbors(row, 'neighbors')
		return Point({'person': np.array(row["style_preference"]), 'neighbors':neighbors}, row["score"])

	@classmethod
	def style_from_neighbor_conv(cls, row):
		neighbors = cls.collect_neighbors(row, 'neighbors')
		return Point(neighbors, row["product.style"])		


	



