
import collections
import random
import pickle
import os.path

import keras
import numpy as np
from keras.preprocessing import text
from keras.utils import np_utils

import neo4j

import experiment
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



	# Applies a per-experiment recipe to Neo4j to get a dataset to train on
	# This performs all transformations in-memory - it is not very efficient
	@staticmethod
	def generate(params):
		Recipe = collections.namedtuple('Recipe', ['params', 'hashing', 'split', 'finalize_x'])

		global_params = QueryParams(golden=params.golden, experiment=params.experiment)
		
		recipes = {
			'review_from_visible_style': Recipe(
					global_params,
					{'style', 'style_preference'},
					lambda row, hashed: Point(np.concatenate((hashed['style_preference'], hashed['style'])), row['score']),
					lambda x: x
				),

			'review_from_hidden_style': Recipe(
					global_params,
					{'style_preference'},
					Dataset.row_transform_review_from_hidden_style,
					lambda x: {'person':np.array([i['person'] for i in x]), 'neighbors': np.array([i['neighbors'] for i in x])}
				)
		}

		return Dataset.execute_recipe(params, recipes[params.experiment])






	@staticmethod
	def execute_recipe(params, recipe):
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

			hashing = Dataset.hash_statement_result(data, recipe.hashing)

			xy = [recipe.split(*i) for i in zip(data, hashing)]

			return Dataset(params, recipe, data, xy)

	@staticmethod
	def lazy_generate(params):

		dataset_file = os.path.join(params.data_dir + '/' + params.experiment + '.pkl')

		if os.path.isfile(dataset_file):
			return pickle.load(open(dataset_file, "rb"))

		else:
			d = Dataset.generate(params)
			pickle.dump(d, open(dataset_file, "wb"))
			return d
		

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
		
		if params.verbose > 0:
			print("Test data sample: ", list(zip(self.test.x, self.test.y))[:10])
		

	# Transforms neo4j results into a hashed version with the same key structure
	# Does not guarantee same hashing scheme used for each column, or for each run
	# TODO: make this all more efficient
	# @argument keys_to_sizes dictionary, the keys of which are the columns that will be hashed, the values of which are the size of each hash space
	@staticmethod
	def hash_statement_result(data:neo4j.v1.StatementResult, keys_to_use:set):

		rows_keyed = [
			{
				key: np.array(x[key])
				for key 
				in keys_to_use
			}
			for x
			in data
		]

		return rows_keyed


	@staticmethod
	def row_transform_review_from_hidden_style(row, hashed):

		other_size = 100

		others = []
		for path in row["others"]:
			other_person = path.nodes[0]
			other_review = path.nodes[1]
			others.append(np.concatenate((
				np.array(other_person.properties['style_preference']),
				[other_review.properties['score']]
			)))

		np.random.shuffle(others)

		if len(others) > other_size:
			others = others[:other_size]

		others = np.pad(others, ((0,0), (1,0)), 'constant', constant_values=1.0) # add 'none' flag

		# pad out if too small
		# note if there are zero others, this won't know the width to make the zeros, so it'll be 1 wide and broadcast later
		if len(others) < other_size:
			delta = other_size - others.shape[0]
			others = np.pad(others, ((0,delta), (0, 0)), 'constant', constant_values=0.0)

		return Point({'person': np.array(row["style_preference"]), 'neighbors':others}, row["score"])


	



