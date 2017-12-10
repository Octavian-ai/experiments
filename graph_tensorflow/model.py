
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

class Model(object):

	@staticmethod
	def generate(params):
		
		input_shape = (1000, 300, 300, 1)
		num_classes = 2

		model = Sequential()
		model.add(Conv2D(32, kernel_size=(3, 3),
						 activation='relu',
						 input_shape=input_shape))
		model.add(Conv2D(64, (3, 3), activation='relu'))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.25))
		model.add(Flatten())
		model.add(Dense(128, activation='relu'))
		model.add(Dropout(0.5))
		model.add(Dense(num_classes, activation='softmax'))

		return model