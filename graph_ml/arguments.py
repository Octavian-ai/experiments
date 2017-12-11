
import argparse

class Arguments(object):
	def parse():

		parser = argparse.ArgumentParser()

		parser.add_argument('--experiment', type=str, default="simple")

		parser.add_argument('--batch_size', type=int, default=16)
		parser.add_argument('--epochs', type=int, default=50)
		parser.add_argument('--random-seed', type=int, default=None)
		parser.add_argument('--verbose', type=int, default=0)

		parser.add_argument('--golden', action='store_true')

		parser.add_argument('--output-dir', type=str, default="./output")
		parser.add_argument('--data-dir', type=str, default="./data")

		return parser.parse_args()
