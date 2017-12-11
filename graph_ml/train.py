
import keras
import numpy as np

from .model import Model
from .dataset import Dataset

class Train(object):

	@staticmethod
	def run(params):

		dataset = Dataset.generate(params)
		model = Model.generate(params, dataset)

		model.fit(np.array(dataset.train.x), np.array(dataset.train.y),
			batch_size=params.batch_size,
			epochs=params.epochs,
			verbose=1,
			validation_data=(np.array(dataset.validate.x), np.array(dataset.validate.y)))

		score = model.evaluate(np.array(dataset.test.x), np.array(dataset.test.y), verbose=0)

		return score