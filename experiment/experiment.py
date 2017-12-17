
from datetime import datetime
from colorama import init, Fore, Style
import logging
import coloredlogs

from graph_ml import Train, Dataset
from .arguments import Arguments
from .directory import directory


init()
logger = logging.getLogger(__name__)


class Experiment(object):
	def __init__(self, name, header, params):
		self.name = name
		self.header = header
		self.params = params
		self.run_tag = str(datetime.now())

	@classmethod
	def run(cls):

		params = Arguments.parse()
		experiment = Experiment(params.experiment, directory[params.experiment], params)

		print(Fore.GREEN)
		print("#######################################################################")
		print(f"📟  Running experiment {experiment.name} {experiment.run_tag}")
		print("#######################################################################")
		print(Style.RESET_ALL)

		if params.verbose > 0:
			coloredlogs.install(level='INFO')
		
		logging.info("Get data")

		dataset = Dataset.get(experiment)

		score = Train.run(experiment, dataset)

		print('Test loss:', score[0])
		print('Test accuracy:', score[1])
		
