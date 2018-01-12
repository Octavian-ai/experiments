
from datetime import datetime
from colorama import init, Fore, Style
import logging
import coloredlogs
import colored_traceback.auto
import os

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
		
		if params.verbose > 0:
			coloredlogs.install(level='INFO', logger=logging.getLogger("experiment"))
			coloredlogs.install(level='INFO', logger=logging.getLogger("graph_ml"))
			coloredlogs.install(level='INFO', logger=logging.getLogger("graph_io"))

		experiment = Experiment(params.experiment, directory[params.experiment], params)

		print(Fore.GREEN)
		print("#######################################################################")
		print(f"📟  Running experiment {experiment.name} {experiment.run_tag}")
		print("#######################################################################")
		print(Style.RESET_ALL)

		dataset = Dataset.get(experiment)
		score = Train.run(experiment, dataset)

		print(Fore.YELLOW)
		print("#######################################################################")
		print("Experiment results")
		print(f"{experiment.name} test loss {score[0]}")
		print(f"{experiment.name} test accuracy {score[1]}")
		print("#######################################################################")
		print(Style.RESET_ALL)

		# t = '-title {!r}'.format(title)
		# s = '-subtitle {!r}'.format(subtitle)
		# m = '-message {!r}'.format(message)
		os.system(f"terminal-notifier -message 'test accuracy {round(score[1]*100)}%' -title Octavian")
		os.system(f"say test accuracy {round(score[1]*100)} percent")
		
