
from datetime import datetime

from graph_ml import Train, Dataset
from .arguments import Arguments
from .directory import directory




class Experiment(object):
	def __init__(self, header, params):
		self.header = header
		self.params = params
		self.run_tag = str(datetime.now())

	@classmethod
	def run(cls):

		params = Arguments.parse()

		experiment = Experiment(directory[params.experiment], params)

		if params.verbose > 0:
			print("Get data")

		dataset = Dataset.get(experiment)

		score = Train.run(experiment, dataset)

		print('Test loss:', score[0])
		print('Test accuracy:', score[1])
		
