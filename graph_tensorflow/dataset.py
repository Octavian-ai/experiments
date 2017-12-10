
from graph_io import SimpleNodeClient

class Dataset(object):

	@staticmethod
	def generate(params):

		if params.graph == 'test':
			return {
				train: {x: [1], y:[1]},
				validate: {x: [1], y:[1]},
				test: {x: [1], y:[1]}
			}

		elif params.graph == 'simple':
			with SimpleNodeClient() as client:
				return client.execute_cypher("HELLO")
		



