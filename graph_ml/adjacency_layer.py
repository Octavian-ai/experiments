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


def cartesian_product_matrix(a, b):
	tile_a = tf.tile(tf.expand_dims(a, 1), [1, tf.shape(b)[0], 1])
	tile_a = tf.expand_dims(tile_a, 2)

	tile_b = tf.tile(tf.expand_dims(b, 0), [tf.shape(a)[0], 1, 1])
	tile_b = tf.expand_dims(tile_b, 2)

	cartesian_product = tf.concat([tile_a, tile_b], axis=2)

	return cartesian_product




class Adjacency(Layer):

	def __init__(self, person_count, product_count, style_width, **kwargs):

		self.use_legacy_1 = True
		self.use_legacy_2 = True

		self.person_count = person_count
		self.product_count = product_count
		self.style_width = style_width

		# if self.use_legacy_1:
		self.dense1 = layers.Dense(units=(style_width), activation=activations.softplus, use_bias=False, kernel_regularizer=Clip)
	
		# if self.use_legacy_2
		self.dense3 = layers.Dense(units=1, activation=partial(activations.relu, alpha=0.1), use_bias=False, kernel_regularizer=Clip)
		
		super(Adjacency, self).__init__(**kwargs)

	def jitter_weights(self, idx_to_jitter, radius=0.2):
		wts = self.get_weights()

		wts = [
			np.array(val + np.random.normal(0, radius, val.shape), dtype=np.float32) if idx in idx_to_jitter else val
			for idx, val in enumerate(wts)
		]
		self.set_weights(wts)

	def __call__(self, inputs, **kwargs):
		
		product_ct = inputs.shape[1]
		person_ct = inputs.shape[2]
		my_batch = product_ct * person_ct

		self.inner_input = Input(batch_shape=(product_ct, person_ct, 2, self.style_width), dtype='float32', name="inner_d0")
		self.reshaped_to_look_like_a_batch = K.reshape(self.inner_input, (product_ct * person_ct, 2 * self.style_width))
		self.dense1_called = self.dense1(self.reshaped_to_look_like_a_batch)
		self.dense3_called = self.dense3(self.dense1_called)
		self.reshaped_to_look_like_adj_mat = K.reshape(self.dense3_called, (product_ct, person_ct, 1))
		
		return super(Adjacency, self).__call__(inputs, **kwargs)


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

		if not self.use_legacy_1:
			# self.dense1 = layers.Dense(units=(style_width), activation=activations.softplus, use_bias=False, kernel_regularizer=Clip)
			self.w1 = self.add_weight(name='w1', 
				shape=(2 * self.style_width, 
						self.style_width),
				initializer='ones',
				regularizer=Clip(),
				trainable=True)


		if not self.use_legacy_2:
			# self.dense3 = layers.Dense(units=1, activation=partial(activations.relu, alpha=0.1), use_bias=False, kernel_regularizer=Clip)
			self.w2 = self.add_weight(name='w2', 
				shape=(self.style_width, 1),
				initializer='ones', # glorot_uniform
				trainable=True,
				regularizer=Clip())


		super(Adjacency, self).build(input_shape)  # Be sure to call this somewhere!

	def call(self, x):
		pr = self.product
		pe = self.person

		self.jitter_weights(idx_to_jitter=[0, 1])

		all_pairs = cartesian_product_matrix(pr, pe)

		flat = K.reshape(all_pairs, (self.product_count * self.person_count, 2 * self.style_width))


		if self.use_legacy_1:
			m = self.dense1.call(flat)
		else:
			m = K.dot(flat, self.w1)
			m = K.softplus(m)

		if self.use_legacy_2:
			m = self.dense3.call(m)
		else:
			m = K.dot(m, self.w2)
			m = K.relu(m, alpha=0.1)


		square = K.reshape(m, (1, self.product_count, self.person_count))
		batched = K.tile(square, [self.batch_size,1,1])
		#proj = K.dot(pr, K.transpose(pe))# + self.noise

		mul = batched * x
		# mul = mul * self.w1 + self.b1
		# mul = K.sigmoid(mul)
		# mul = mul * self.w2 + self.b2

		return mul

	def compute_output_shape(self, input_shape):
		return input_shape


