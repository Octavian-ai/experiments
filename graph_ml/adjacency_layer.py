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
		self.person = self.add_weight(name='people', 
			shape=(self.person_count, self.style_width),
			initializer='uniform',
			trainable=True)

		self.product = self.add_weight(name='product', 
			shape=(self.product_count, self.style_width),
			initializer='uniform',
			trainable=True)

		self.w1 = self.add_weight(name='w1', 
			shape=(1,),
			initializer='one',
			trainable=True)

		self.b1 = self.add_weight(name='b1', 
			shape=(1,),
			initializer='zero',
			trainable=True)

		self.w2 = self.add_weight(name='w2', 
			shape=(1,),
			initializer='one',
			trainable=True)

		self.b2 = self.add_weight(name='b2', 
			shape=(1,),
			initializer='zero',
			trainable=True)

		super(Adjacency, self).build(input_shape)  # Be sure to call this somewhere!

	def call(self, x):
		proj = K.dot(self.product, K.transpose(self.person))
		mul = proj * x
		mul = mul * self.w1 + self.b1
		mul = K.sigmoid(mul)
		mul = mul * self.w2 + self.b2

		return mul

	def compute_output_shape(self, input_shape):
		return input_shape