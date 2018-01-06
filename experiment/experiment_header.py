
from typing import List
from graph_io.classes import DatasetName

class ExperimentHeader(object):
	def __init__(self, doc="", dataset_name: DatasetName=None, cypher_query=None, target=None, params={}, lazy_params:List[str]=[]):
		# Jesus I have to spell this out?!
		# WTF are the python language devs doing?!
		self.dataset_name = dataset_name
		self.doc = doc
		self.cypher_query = cypher_query
		self.target = target
		self.params = params
		self.lazy_params = lazy_params