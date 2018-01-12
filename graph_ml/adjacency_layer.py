from keras import backend as K
from keras.engine.topology import Layer
import numpy as np

class Adjacency(Layer):

	def __init__(self, person_count, product_count, style_width, **kwargs):
		self.person_count = person_count
		self.product_count = product_count
		self.style_width = style_width

		super(Adjacency, self).__init__(**kwargs)

	def build(self, input_shape):
		# Create a trainable weight variable for this layer.
		self.person_var = self.add_weight(name='people', 
			shape=(self.person_count, self.style_width),
			initializer='uniform',
			trainable=True)

		self.product_var = self.add_weight(name='product', 
			shape=(self.product_count, self.style_width),
			initializer='uniform',
			trainable=True)

		super(Adjacency, self).build(input_shape)  # Be sure to call this somewhere!

	def call(self, x):
		# print("x ", x.shape)
		proj = K.dot(self.product_var, K.transpose(self.person_var))
		# print("proj ", proj.shape)
		mul = proj * x
		# print("mul", mul.shape)
		return mul

	def compute_output_shape(self, input_shape):
		return input_shape