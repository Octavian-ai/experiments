
import argparse

class Arguments(object):
	def parse():

		parser = argparse.ArgumentParser()

		parser.add_argument('--experiment', type=str, default="simple")

		parser.add_argument('--batch_size', type=int, default=100)
		parser.add_argument('--epochs', type=int, default=5)

		parser.add_argument('--output-dir', type=str, default="./output")
		parser.add_argument('--data-dir', type=str, default="./data")

		return parser.parse_args()