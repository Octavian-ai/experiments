from keras import backend as K
from keras.engine.topology import Layer
from keras import regularizers, initializers
import numpy as np

class PD(regularizers.Regularizer):
	def __init__(self, a=0.0001, b=0.0, axis=-1):
		self.a = K.cast_to_floatx(a)
		self.b = K.cast_to_floatx(b)

		self.axis = axis

	def __call__(self, x):
		sum_to_one = K.abs(1.0 - K.sum(K.abs(x), axis=self.axis))
		different_by_one = K.abs(1.0 - K.abs(x[:,0] - x[:,1]))
		core =  self.a * sum_to_one + self.b * different_by_one

		return K.sum(core)

	def get_config(self):
		return {'a': float(self.a), 'b': float(self.b)}

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
			initializer=initializers.RandomUniform(minval=0, maxval=1),
			# initializer='ones',
			# regularizer=PD(),
			trainable=True)

		self.product = self.add_weight(name='product', 
			shape=(self.product_count, self.style_width),
			initializer=initializers.RandomUniform(minval=0, maxval=1),
			# initializer='ones',
			# regularizer=PD(),
			trainable=True)

		# self.w1 = self.add_weight(name='w1', 
		# 	shape=(1,),
		# 	initializer='one',
		# 	trainable=True)

		# self.b1 = self.add_weight(name='b1', 
		# 	shape=(1,),
		# 	initializer='zero',
		# 	trainable=True)

		# self.w2 = self.add_weight(name='w2', 
		# 	shape=(1,),
		# 	initializer='one',
		# 	trainable=True)

		# self.b2 = self.add_weight(name='b2', 
		# 	shape=(1,),
		# 	initializer='zero',
		# 	trainable=True)

		super(Adjacency, self).build(input_shape)  # Be sure to call this somewhere!

	def call(self, x):
		# pr = K.softmax(self.product)
		# pe = K.softmax(self.person)

		pr = K.concatenate([
			self.product, 
			1.0 - self.product
		], axis=1)
		
		pe = K.concatenate([
			self.person, 
			1.0 - self.person
		], axis=1)

		# proj = K.dot(self.product, K.transpose(self.person))
		proj = K.dot(pr, K.transpose(pe))
		
		mul = proj * x
		# mul = mul * self.w1 + self.b1
		# mul = K.sigmoid(mul)
		# mul = mul * self.w2 + self.b2

		return mul

	def compute_output_shape(self, input_shape):
		return input_shape