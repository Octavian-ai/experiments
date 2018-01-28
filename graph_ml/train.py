
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


class SpecialValidator(keras.callbacks.Callback):
	def __init__(self, experiment, dataset, model):
		self.experiment = experiment
		self.model = model
		self.dataset = dataset
		super(keras.callbacks.Callback, self).__init__()

	
	def on_train_end(self, logs):
		self.test()

	def on_epoch_end(self, epoch, logs):
		self.test()

	def test(self):
		print() # Clear from epoch status bar
		for (label, genie) in self.dataset.generator.items():
			# print(f"Prediction for {label}")

			row = genie.peek()
			y_true = row[1][0]
			x_test = row[0][0]

			y_pred = self.model.predict_generator(
				generator=genie,
				steps=1,
				workers=0,
				use_multiprocessing=False,
			)
			y_pred = np.array(y_pred[0])

			y_correct = np.isclose(y_pred, y_true, atol=0.1)
			y_zero = np.isclose(y_pred, 0, atol=0.1)
			
			# The bits that should be one
			y_true_set_and_in_mask = np.where(np.greater(y_true, 0.1), np.greater(x_test, 0.1), False)
			
			# The bits that should be one and were one
			y_masked = np.where(y_true_set_and_in_mask, y_correct, False)
			
			# The correct predictions for the input adj
			y_masked_david = np.where(np.greater(x_test, 0.1), y_correct, False)

			# print("x_test: ",x_test)
			# print("y_true: ", y_true)
			# print("y_pred: ", np.around(y_pred, 1))
			# print("y_correct: ",y_correct)
			# print(f"y_masked {np.count_nonzero(y_masked)} / {np.count_nonzero(y_correct)} / {np.count_nonzero(x_test)}")
			
			net_accuracy = round(np.count_nonzero(y_masked) / (np.count_nonzero(y_true_set_and_in_mask)+0.001) * 100, 3)
			net_accuracy_david = round(np.count_nonzero(y_masked_david) / (np.count_nonzero(x_test)+0.001) * 100, 3)
			gross_accuracy = round(np.count_nonzero(y_correct) / np.size(y_correct) * 100, 3)

			print(f"{label} 1-accuracy: {net_accuracy}%  accuracy: {net_accuracy_david}%")
			# print()





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
			#StopEarlyIfAbove(verbose=params.verbose),
			SpecialValidator(experiment, dataset, model),
			# keras.callbacks.ModelCheckpoint(params_file, verbose=params.verbose, save_best_only=True, monitor='val_loss', mode='auto', period=3),
			# keras.callbacks.TensorBoard(log_dir=generate_output_path(experiment, f"_log/{experiment.run_tag}/")),
			#keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.000000001, patience=8, verbose=0, mode='auto')
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


		if params.print_weights:
			for layer in model.layers:
				for var, weight in zip(layer.weights, layer.get_weights()):
					print(f"{var.name} {np.around(weight, decimals=2)}")


		return score


