
import keras

from .model import Model
from .dataset import Dataset

class Train(object):

	@staticmethod
	def run(params):

		dataset = Dataset.generate(params)
		# model = Model.generate(params, dataset)

		# model.compile(loss=keras.losses.categorical_crossentropy,
		# 			  optimizer=keras.optimizers.Adadelta(),
		# 			  metrics=['accuracy'])

		# model.fit(dataset.train.x, dataset.train.y,
		# 		  batch_size=params.batch_size,
		# 		  epochs=params.epochs,
		# 		  verbose=1,
		# 		  validation_data=(dataset.validate.x, dataset.validate.y))
		# score = model.evaluate(dataset.test.x, dataset.test.y, verbose=0)

		# return score
		return [0,0]