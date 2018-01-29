

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

RecordGenerator = Generator[neo4j.v1.Record, None, None]
PointGenerator = Generator[Point, None, None]

class Recipe:
	def __init__(self, 
		transform:Callable[[RecordGenerator], PointGenerator] = None,
		query:Callable[[], RecordGenerator] = None,
		partition:Callable[[PointGenerator], Generator[Tuple[str, Point], None, None]] = None,
		split:Callable[[neo4j.v1.Record], Point] = None,
		finalize_x = None):

		self.transform = transform
		self.query = query
		self.partition = partition

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

		if partition is None:
			def default_partition(data):
				random.shuffle(data)
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
			self.partition = default_partition


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

		assert len(arr) == length, f"ensure_length failed to resize, {len(arr)} != {length}"

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
	def review_from_all_hidden_simple_unroll(cls, experiment):
		def t(row):
			length = experiment.header.params["neighbor_count"]
			neighbors = np.array(row["neighbors"])
			delta = length - neighbors.shape[0]

			if delta > 0:
				neighbors = np.pad(neighbors, ((0,delta), (0, 0)), 'constant', constant_values=0.0)
			
			return Point(neighbors, row["score"])

		return Recipe(t)

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
		bs = experiment.params.batch_size
		person_product = {}

		reviews_per_person = Counter()
		reviews_per_product = Counter()

		pr_c = experiment.header.params["product_count"]
		pe_c = experiment.header.params["person_count"]

		shape = (pr_c, pe_c)
		unmasked_products=np.zeros(shape=(pr_c,))
		unmasked_products[0] = 1
		unmasked_people=np.zeros(shape=(pe_c,))
		cache = []
		training_mask = np.zeros(shape)
		pause=[0]
		def gen_output(datas):
			for i in range(bs * experiment.header.params["batch_per_epoch"]):
				for partition, pt in datas.items():
					if partition=="train":
						mask_seed = np.random.randint(100, size=shape)
						training_rand = np.greater(mask_seed, 50)
						pt = Point(np.where(training_rand, pt.x, 0), np.where(training_rand, pt.y, 0))
					if partition=="train" and False:
						pe_flag = False
						pr_flag = False
						if pause[0] > 256:

							def do_product():
								if not pr_flag:
									for x in range(pe_c):
										if unmasked_people[x] == 0 and any(pt.x[y][x] == 1 for y in range(pr_c) if unmasked_products[y] == 1):
											unmasked_people[x] = 1
											pe_flag = True
											break

							def do_person():
								if not pe_flag:
									for y in range(pr_c):
										if unmasked_products[y] == 0 and any(pt.x[y][x] == 1 for x in range(pe_c) if unmasked_people[x] == 1):
											unmasked_products[y] = 1
											pr_flag = True
											break

							if random.random() > 0.5:
								do_product()
							else:
								do_person()

							if not pr_flag and not pe_flag:
								for x in range(pe_c):
									if unmasked_people[x] == 0:
										unmasked_people[x] = 1
										pe_flag = True
										break
								if not pe_flag:
									for y in range(pr_c):
										if unmasked_products[y] == 0:
											unmasked_products[y] = 1
											pr_flag = True
											break
							for x in range(pe_c):
								#TODO this is like a np.cross or something
								for y in range(pr_c):
									if unmasked_people[x] * unmasked_products[y] == 1:
										training_mask[y][x] = 1
							if not pe_flag and not pr_flag:
								assert np.sum(training_mask) == pr_c * pe_c
								print('all data')
							pause[0] = 0
						pause[0]+=1

						pt = Point(np.where(training_mask, pt.x, 0), np.where(training_mask, pt.y, 0))
						#print(np.sum(pt.x))
						#print(np.sum(pt.y))
					yield (partition, pt)
				# yield Point(adj_con, adj_score)

		def transform(stream):
			if len(cache) == 1:
				return  gen_output(cache[0])

			data = list(stream)

			products = set()
			people = set()
			# Construct adjacency dict
			for i in data:
				if i["person_id"] not in person_product:
					person_product[i["person_id"]] = {}

				if len(people) < pe_c or i["person_id"] in people:
					if len(products) < pr_c or i["product_id"] in products:

						person_product[i["person_id"]][i["product_id"]] = i["score"]

						reviews_per_person[i["person_id"]] += 1
						reviews_per_product[i["product_id"]] += 1

						products.add(i["product_id"])
						people.add(i["person_id"])

			def exists(person, product):
				return 1.0 if person in person_product and product in person_product[person] else 0.0

			def score(person, product):
				return person_product.get(person, 0.0).get(product, 0.0) 

			ppe = list(dict(reviews_per_person).values())
			ppr = list(dict(reviews_per_product).values())

			#print("Reviews per product: ", np.histogram(ppe) )
			#print("Reviews per person: ", np.histogram(ppr) )

			#logger.info(f"People returned {len(people)} of capacity {pe_c}")
			#logger.info(f"Products returned {len(products)} of capacity {pr_c}")

			people   = sorted(list(people))[:pe_c]
			products = sorted(list(products))[:pr_c]

			def build(fn):
				return DatasetHelpers.ensure_length(np.array([
					DatasetHelpers.ensure_length(
						np.array([fn(person, product) for person in people])
					, pe_c) for product in products
				]), pr_c)

			adj_score = build(score)
			adj_con = build(exists)

			# print("Connections:",adj_con)
			# print("Scores:",adj_score)

			assert_mtx_shape(adj_score, shape, "adj_score")
			assert_mtx_shape(adj_con,   shape)

			mask_seed = np.random.randint(75, size=shape)
			masks = {
				"test":     np.equal(mask_seed, 0),
				"train":    np.greater(mask_seed, 1),
				"validate": np.equal(mask_seed, 1),
				"all":      Point(adj_con, adj_score)
			}

			def gen_d(mask):
				return Point(np.where(mask, adj_con, 0), np.where(mask, adj_score, 0))

			datas = {
				k: gen_d(v)
				for (k, v) in masks.items()
			}

			#for k, v in datas.items():
			#	print(k, np.sum(v.x), np.sum(v.y), 1.0 - np.sum(v.y)/np.prod(shape), np.sum(np.multiply(v.x, v.y)))

			# print(datas)
			cache.append(datas)

			return gen_output(datas)


		return Recipe(transform=transform, partition=lambda x:x)


