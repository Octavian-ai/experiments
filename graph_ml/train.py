
import keras
import numpy as np
from keras.callbacks import Callback

from .model import Model
from .dataset import Dataset

class StopEarlyIfAbove(Callback):
	def __init__(self, monitor='val_acc', value=0.99, verbose=0):
		super(Callback, self).__init__()
		self.monitor = monitor
		self.value = value
		self.verbose = verbose
		self.stopped_epoch = 0

	def on_epoch_end(self, epoch, logs={}):
		current = logs.get(self.monitor)
		if current is None:
			print("Early stopping requires %s available!" % self.monitor)
			exit()

		if current > self.value:
			self.stopped_epoch = epoch
			self.model.stop_training = True

	def on_train_end(self, logs=None):
		if self.stopped_epoch > 0 and self.verbose > 0:
			print("Epoch {}: early stopping {} > {}".format(self.stopped_epoch+1, self.monitor, self.value))




class Train(object):

	@staticmethod
	def run(params):

		if params.random_seed is not None:
			np.random.seed(params.random_seed)

		dataset = Dataset.lazy_generate(params)
		model = Model.generate(params, dataset)

		callbacks = [
			StopEarlyIfAbove(verbose=params.verbose)
		]
		
		model.fit(dataset.train.x, dataset.train.y,
			batch_size=params.batch_size,
			epochs=params.epochs,
			verbose=params.verbose,
			validation_split=0.1,
			shuffle=True,
			callbacks=callbacks
		)

		score = model.evaluate(dataset.test.x, dataset.test.y, verbose=0)

		if score[1] < 1.0 and params.verbose > 1:
			for layer in model.layers:
				print("Layer weights", layer.get_weights())

		return score


