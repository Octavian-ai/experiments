
import collections
import random

import keras
import numpy as np
from keras.preprocessing import text
from keras.utils import np_utils

import neo4j

from graph_io import *

Recipe = collections.namedtuple('Recipe', ['query', 'hashing', 'split'])

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

	def __init__(self, params, data, xy):
		self.params = params
		self.data = data
		self.data_xy = xy

		self.input_shape = (len(xy[0].x),)


		## Split data up into train/validate/test
		self.train = Point([], [])
		self.validate = Point([], [])
		self.test = Point([], [])

		random.seed(params.random_seed)

		def store_datum(i):
			r = random.random()
			if r > 0.9:
				self.test.append(i)
			elif r > 0.8:
				self.validate.append(i)
			else:
				self.train.append(i)

		for i in self.data_xy:
			store_datum(i)

		# print("Train", self.train)
		# print("validate", self.validate)
		print("test", self.test)


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
			k: np_utils.to_categorical( [text.one_hot(str(i[k]),n) for i in data] )
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



	# Applies a per-experiment recipe to Neo4j to get a dataset to train on
	@staticmethod
	def generate(params):
		with SimpleNodeClient() as client:
			query_params = QueryParams()

			recipies = {
				'simple': Recipe(
						"""MATCH p=
								(a:PERSON {is_golden:false}) 
									-[:WROTE {is_golden:false}]-> 
								(b:REVIEW {is_golden:false}) 
									-[:OF {is_golden:false}]-> 
								(c:PRODUCT {is_golden:false})
							RETURN a.style_preference AS preference, c.style AS style, b.score AS score
							LIMIT 10000000
						""",
						{'preference':4, 'style':4},
						lambda row, hashed: Point(np.concatenate((hashed['preference'], hashed['style'])), row['score'])
					)
			}

			recipe = recipies[params.experiment]
			data = client.execute_cypher(CypherQuery(recipe.query), query_params)
			data = list(data) # so we can do a few passes
			hashing = Dataset.hash_statement_result(data, recipe.hashing)

			print("Data", data[:10])

			# Once I get my shit together,
			# 1) use iterators
			# 2) move to streaming
			# 3) move to hdf5
			xy = [recipe.split(*i) for i in zip(data, hashing)]

			return Dataset(params, data, xy)
		



