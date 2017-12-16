
if __name__ == '__main__':
	from graph_io import SimpleNodeClient
	import argparse

	parser = argparse.ArgumentParser()
	parser.add_argument('--experiment', type=str, default="1")
	FLAGS = parser.parse_args()

	with SimpleNodeClient() as client:

		# Just give me a fucking switch statement you imperative piece of shit
		if FLAGS.experiment == '1':
			# Let's move these to a dict for my factory
			from data_sets.synthetic_review_prediction.experiment_1 import run
		elif FLAGS.experiment == '2':
			from data_sets.synthetic_review_prediction.experiment_2 import run
		elif FLAGS.experiment == '3':
			from data_sets.synthetic_review_prediction.experiment_3 import run
		elif FLAGS.experiment == '4':
			from data_sets.synthetic_review_prediction.experiment_4 import run
			
		run(client)