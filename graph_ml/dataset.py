
import collections

from graph_io import *

class Dataset(object):

	def __init__(self, data, x, y):
		self.data = data
		self.data_x = x
		self.data_y = y

		for r in self.data_x:
			print(r)

		self.input_shape = (1,2,3,4)

		self.train = {}
		self.test = {}
		self.validate = {}


	@staticmethod
	def generate(params):
		with SimpleNodeClient() as client:
			query_params = QueryParams()

			Recipe = collections.namedtuple('Recipe', ['query', 'get_x', 'get_y'])

			recipies = {
				'movie': Recipe(
						"MATCH p=(a)-[r:ACTS_IN]->(b)<-[q:RATED]-(c) RETURN a.name, q.stars LIMIT 100",
						lambda row: (row.name),
						lambda row: row.stars
					),
				
				'simple': Recipe(
						"""MATCH p=
								(a:PERSON {isGolden:false}) 
									-[:WROTE {isGolden:false}]-> 
								(b:REVIEW {isGolden:false}) 
									-[:OF {isGolden:false}]-> 
								(c:PRODUCT {isGolden:false})
							RETURN a.style_preference, c.style, b.score
							LIMIT 1000
						""",
						lambda row: (row.style_preference, row.style), 
						lambda row: row.score
					)
			}

			recipe = recipies[params.experiment]
			data = client.execute_cypher(CypherQuery(recipe.query), query_params)
			x = map(recipe.get_x, data)
			y = map(recipe.get_y, data)

			return Dataset(data, x, y)
		



