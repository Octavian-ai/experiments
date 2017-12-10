
import keras

from model import Model
from dataset import Dataset

class Train(object):

	@staticmethod
	def run(params):

		data = Dataset.generate(params)
		model = Model.generate(params)

		model.compile(loss=keras.losses.categorical_crossentropy,
					  optimizer=keras.optimizers.Adadelta(),
					  metrics=['accuracy'])

		model.fit(data.train.x, data.train.y,
				  batch_size=params.batch_size,
				  epochs=params.epochs,
				  verbose=1,
				  validation_data=(data.validate.x, data.validate.y))
		score = model.evaluate(data.text.x, data.tex.y, verbose=0)

		return score



if __name__ == '__main__':

	params = Arguments.parse()
	score = Train.run(params)

	print('Test loss:', score[0])
	print('Test accuracy:', score[1])