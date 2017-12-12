
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
		with SimpleNodeClient() as client:

			Recipe = collections.namedtuple('Recipe', ['params', 'hashing', 'split'])

			global_params = QueryParams(golden=params.golden, experiment=params.experiment)
			
			recipes = {
				'review_from_visible_style': Recipe(
						global_params,
						{'preference':4, 'style':4},
						lambda row, hashed: Point(np.concatenate((hashed['preference'], hashed['style'])), row['score'])
					),

				'review_from_hidden_style': Recipe(
						global_params,
						{'preference':4},
						lambda row, hashed: Point((hashed['preference']), row['score'])
					)
			}

			return Dataset.execute_recipe(params, recipes[params.experiment])






	@staticmethod
	def execute_recipe(params, recipe):
		data = client.execute_cypher(CypherQuery(experiment.directory[params.experiment].cypher_query), recipe.params)

		# Once I get my shit together,
		# 1) use iterators
		# 2) move to streaming
		# 3) move to hdf5

		data = list(data) # so we can do a few passes

		if len(data) == 0:
			raise Exception('Neo4j query returned no data, cannot train the network') 

		if params.verbose > 0:
			print("Retrieved {} rows from Neo4j".format(len(data)))
			print("Data sample: ", data[:10])

		hashing = Dataset.hash_statement_result(data, recipe.hashing)

		xy = [recipe.split(*i) for i in zip(data, hashing)]

		return Dataset(params, data, xy)

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
	def __init__(self, params, data, xy):
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
		self.train.x = np.array(self.train.x)
		self.train.y = np.array(self.train.y)

		self.test.x = np.array(self.test.x)
		self.test.y = np.array(self.test.y)
		
		if params.verbose > 0:
			print("Test data sample: ", list(zip(self.test.x, self.test.y))[:10])
		

	# Transforms neo4j results into a hashed version with the same key structure
	# Does not guarantee same hashing scheme used for each column, or for each run
	# TODO: make this all more efficient
	# @argument keys_to_sizes dictionary, the keys of which are the columns that will be hashed, the values of which are the size of each hash space
	@staticmethod
	def hash_statement_result(data:neo4j.v1.StatementResult, keys_to_sizes:dict):

		a = {
			k: [str(i[k])for i in data]
			for (k,n) 
			in keys_to_sizes.items()
		}

		b = {
			k: [text.one_hot(str(i[k]),n) for i in data]
			for (k,n) 
			in keys_to_sizes.items()
		}

		columns_as_one_hot = {
			k: np_utils.to_categorical( np.unique([i[k] for i in data], return_inverse=True)[1] )
			for (k,n) 
			in keys_to_sizes.items()
		}

		rows_keyed = [
			{
				key: columns_as_one_hot[key][i]
				for key 
				in keys_to_sizes.keys()
			}

			for i in range(len(data))
		]

		return rows_keyed



	



