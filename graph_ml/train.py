
import os.path
from datetime import datetime
import logging
from sklearn.metrics import classification_report
import itertools

import keras
import numpy as np
import keras.callbacks

from .model import Model
from .dataset import Dataset
from .path import generate_output_path

logger = logging.getLogger(__name__)

class StopEarlyIfAbove(keras.callbacks.Callback):
	def __init__(self, monitor='val_acc', value=0.99, verbose=0, patience=3):
		super(keras.callbacks.Callback, self).__init__()
		self.monitor = monitor
		self.value = value
		self.verbose = verbose
		self.stopped_epoch = 0
		self.patience = patience

	def on_epoch_end(self, epoch, logs={}):
		current = logs.get(self.monitor)
		if current is None:
			logger.error("Early stopping requires %s available!" % self.monitor)
			exit()

		if current > self.value:
			self.patience -= 1
			if self.patience <= 0:
				self.stopped_epoch = epoch
				self.model.stop_training = True

	def on_train_end(self, logs=None):
		if self.stopped_epoch > 0 and self.verbose > 0:
			logger.info("Epoch {}: early stopping {} > {}".format(self.stopped_epoch+1, self.monitor, self.value))


class TraceCallback(keras.callbacks.Callback):
	def __init__(self, ):
		super(keras.callbacks.Callback, self).__init__()

	def on_epoch_start(self, epoch, logs={}):
		logger.info(f"Epoch {epoch} start")

	def on_epoch_end(self, epoch, logs={}):
		logger.info("Epoch end")

	def on_train_end(self, logs=None):
		logger.info("Train end")

	# def on_batch_end(self, batch_index, logs):
	# 	logger.info("Batch end")

	# def on_batch_begin(self, batch_index, logs):
	# 	logger.info("Batch begin " + batch_index)
		


class Train(object):

	@staticmethod
	def run(experiment, dataset):

		params = experiment.params

		if params.random_seed is not None:
			np.random.seed(params.random_seed)

		logger.info("Generate model")

		model = Model.generate(experiment, dataset)
		params_file = generate_output_path(experiment, ".hdf5")

		if os.path.isfile(params_file) and params.load_weights:
			model.load_weights(params_file)

		callbacks = [
			StopEarlyIfAbove(verbose=params.verbose),
			# TraceCallback(),
			# keras.callbacks.ModelCheckpoint(params_file, verbose=params.verbose, save_best_only=True, monitor='val_loss', mode='auto', period=3),
			# keras.callbacks.TensorBoard(log_dir=generate_output_path(experiment, f"_log/{experiment.run_tag}/")),
			# keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=4, verbose=0, mode='auto')
		]

		# TODO: move to more general overriding mechanism
		# Perhaps unify os.environ, arguments, experiment parameters
		if params.epochs is not None:
			epochs = params.epochs
		else:
			epochs = experiment.header.params.get('epochs', 20)

		logger.info("Fit model")

		# Once I've worked out Python multithreading conflicts we can introduce workers > 0
		model.fit_generator(
			generator=dataset.train_generator,
			steps_per_epoch=dataset.steps_per_epoch,
			validation_data=dataset.validation_generator,
			validation_steps=dataset.validation_steps,

			epochs=epochs,
			verbose=params.verbose,

			workers=0,
			use_multiprocessing=False,
			shuffle=True,
			callbacks=callbacks
		)

		logger.info("Evaluate model")

		score = model.evaluate_generator(
			generator=dataset.test_generator,
			steps=dataset.test_steps,
			workers=0,
			use_multiprocessing=False,
		)

		generate_classification_report = False
		if generate_classification_report:
			data_test = list(itertools.islice(dataset.test_generator, dataset.test_steps))
			y_test = [i[1] for i in data_test]

			y_pred = model.predict_generator(
				generator=dataset.test_generator,
				steps=dataset.test_steps,
				workers=0,
				use_multiprocessing=False,
			)
			y_pred = list(y_pred)
			# TODO: I need to de-sequence the data to make this work
			print(classification_report(y_test, y_pred))
			

		if params.print_weights:
			for layer in model.layers:
				for var, weight in zip(layer.weights, layer.get_weights()):
					print(f"{var.name} {weight}")

		return score


