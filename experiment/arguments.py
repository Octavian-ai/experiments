
import argparse

from .directory import directory, default_experiment

class Arguments(object):
	def parse():

		parser = argparse.ArgumentParser()

		parser.add_argument('--experiment', type=str, default=default_experiment, choices=directory.keys())
		parser.add_argument('--dataset-name', type=str, default=None)


		parser.add_argument('--batch_size', type=int, default=32)
		parser.add_argument('--epochs', type=int, default=None)
		parser.add_argument('--random-seed', type=int, default=None)
		parser.add_argument('--verbose', type=int, default=1)

		parser.add_argument('--golden', action='store_true')
		parser.add_argument('--not-lazy', dest='lazy', action='store_false')
		parser.add_argument('--no-say', dest='say_result', action='store_false')
		parser.add_argument('--load-weights', action='store_true')
		parser.add_argument('--print-weights', action='store_true')
		parser.add_argument('--custom-test', action='store_true')

		parser.add_argument('--output-dir', type=str, default="./output")
		parser.add_argument('--data-dir', type=str, default="./data")

		return parser.parse_args()
