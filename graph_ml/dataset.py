
from graph_io import *

class Dataset(object):

	def __init__(self, data):
		self.data = data

		self.input_shape = (1,2,3,4)
		
		self.train = {}
		self.test = {}
		self.validate = {}


	@staticmethod
	def generate(params):

		if params.experiment == 'test':
			return {
				train: {x: [1], y:[1]},
				validate: {x: [1], y:[1]},
				test: {x: [1], y:[1]}
			}

		elif params.experiment == 'simple':
			with SimpleNodeClient() as client:

				query = CypherQuery("MATCH p=()-[r:DIRECTED]->()<-[q:ACTS_IN]-() RETURN p LIMIT 1000")
				query_params = QueryParams()

				return Dataset(client.execute_cypher(query, query_params))
		



