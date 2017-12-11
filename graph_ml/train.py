
import keras
import numpy as np

from .model import Model
from .dataset import Dataset

class Train(object):

	@staticmethod
	def run(params):

		np.random.seed(params.random_seed)

		dataset = Dataset.generate(params)
		model = Model.generate(params, dataset)

		model.fit(dataset.train.x, dataset.train.y,
			batch_size=params.batch_size,
			epochs=params.epochs,
			verbose=params.verbose,
			validation_split=0.1,
			shuffle=True)

		score = model.evaluate(dataset.test.x, dataset.test.y, verbose=0)

		return score