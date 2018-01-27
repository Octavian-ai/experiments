from keras import backend as K
import tensorflow as tf
from keras.engine.topology import Layer, Input
from keras import regularizers, initializers, layers, activations
from functools import partial
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


class Clip(regularizers.Regularizer):
	def __init__(self, max=1):
		self.max = max

	def __call__(self, x):
		K.clip(x, min_value=-1, max_value=1)

	def get_config(self):
		return {'max': float(self.max)}



class Adjacency(Layer):

	def __init__(self, person_count, product_count, style_width, **kwargs):

		self.person_count = person_count
		self.product_count = product_count
		self.style_width = style_width

		super(Adjacency, self).__init__(**kwargs)

	def jitter_weights(self, idx_to_jitter, radius=0.2):
		wts = self.get_weights()

		wts = [
			np.array(val + np.random.normal(0, radius, val.shape), dtype=np.float32) if idx in idx_to_jitter else val
			for idx, val in enumerate(wts)
		]
		self.set_weights(wts)

	def cartesian_product_matrix(self, a, b):
		tile_a = tf.tile(tf.expand_dims(a, 1), [1, tf.shape(b)[0], 1])
		tile_a = tf.expand_dims(tile_a, 2)

		tile_b = tf.tile(tf.expand_dims(b, 0), [tf.shape(a)[0], 1, 1])
		tile_b = tf.expand_dims(tile_b, 2)

		cartesian_product = tf.concat([tile_a, tile_b], axis=2)
		return cartesian_product


	def build(self, input_shape):
		self.batch_size = input_shape[0]

		# Create a trainable weight variable for this layer.
		self.person = self.add_weight(name='people', 
			shape=(self.person_count, self.style_width),
			initializer='zero',
			trainable=True)

		self.product = self.add_weight(name='product', 
			shape=(self.product_count, self.style_width),
			initializer='zero',
			trainable=True)

		

		# self.w1 = self.add_weight(name='w1', 
		# 	shape=(2 * self.style_width, 
		# 			self.style_width),
		# 	initializer='glorot_uniform',
		# 	regularizer=Clip(),
		# 	trainable=True)


		# self.w2 = self.add_weight(name='w2', 
		# 	shape=(self.style_width, 1),
		# 	initializer='glorot_uniform', # glorot_uniform
		# 	trainable=True,
		# 	regularizer=Clip())

		super(Adjacency, self).build(input_shape)  # Be sure to call this somewhere!

	def call_dot(self, x): 
		self.jitter_weights(idx_to_jitter=[0, 1])

		proj = K.dot(self.product, K.transpose(self.person))
		mul = proj * x
		
		return mul

	def call_dense(self, x):
		self.jitter_weights(idx_to_jitter=[0, 1, 2, 3])

		all_pairs = self.cartesian_product_matrix(self.product, self.person)
		flat = K.reshape(all_pairs, (self.product_count * self.person_count, 2 * self.style_width))
	
		m = K.dot(flat, self.w1)
		m = K.softplus(m)

		m = K.dot(m, self.w2)
		m = K.relu(m, alpha=0.1)

		m = K.reshape(m, (1, self.product_count, self.person_count))
		mul = m * x

		return mul

	def call(self, x):
		return self.call_dot(x)

	def compute_output_shape(self, input_shape):
		return input_shape


