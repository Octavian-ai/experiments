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
		self.person_count = person_count
		self.product_count = product_count
		self.style_width = style_width
		self.dense1 = layers.Dense(units=(style_width), activation=activations.tanh, use_bias=False)
		self.dense2 = layers.Dense(units=(style_width), activation=activations.tanh, use_bias=False)
		self.dense3 = layers.Dense(units=1, activation=partial(activations.relu, alpha=0.1), use_bias=False)
		super(Adjacency, self).__init__(**kwargs)

	def __call__(self, inputs, **kwargs):
		self.batch_size = inputs.shape[0]
		product_ct = inputs.shape[1]
		person_ct = inputs.shape[2]
		my_batch = product_ct * person_ct

		self.inner_input = Input(batch_shape=(product_ct, person_ct, 2, self.style_width), dtype='float32', name="inner_d0")
		self.reshaped_to_look_like_a_batch = K.reshape(self.inner_input, (product_ct * person_ct, 2 * self.style_width))
		self.dense1_called = self.dense1(self.reshaped_to_look_like_a_batch)
		self.dense2_called = self.dense2(self.dense1_called)
		self.dense3_called = self.dense3(self.dense2_called)
		self.reshaped_to_look_like_adj_mat = K.reshape(self.dense3_called, (product_ct, person_ct, 1))
		return super(Adjacency, self).__call__(inputs, **kwargs)


	def build(self, input_shape):
		# Create a trainable weight variable for this layer.
		self.person = self.add_weight(name='people', 
			shape=(self.person_count, self.style_width),
			initializer=initializers.RandomUniform(minval=0, maxval=0),
			# initializer='ones',
			regularizer=Clip(),
			trainable=True)

		self.product = self.add_weight(name='product', 
			shape=(self.product_count, self.style_width),
			initializer=initializers.RandomUniform(minval=0, maxval=0),
			# initializer='ones',
			regularizer=Clip(),
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
		pr = self.product
		pe = self.person


		#pr = K.concatenate([
		#	self.product,
		#	1.0 - self.product
		#], axis=1)
		
		#pe = K.concatenate([
		#	self.person,
		#	1.0 - self.person
		#], axis=1)

		# proj = K.dot(self.product, K.transpose(self.person))

		#pr += (1 - pr) * K.random_normal(shape=K.shape(pr),
		#					 mean=0,
		#					 stddev=0.2)

		#pe += (1 - pe) * K.random_normal(shape=K.shape(pe),
		#					 mean=0,
		#					 stddev=0.2)

		wts = self.get_weights()
		temp_pe = wts[0] + np.random.normal(0, 0.001, wts[0].shape)
		temp_pr = wts[1] + np.random.normal(0, 0.001, wts[1].shape)
		self.set_weights([np.array(temp_pe, dtype=np.float32), np.array(temp_pr, dtype=np.float32)])

		all_pairs = cartesian_product_matrix(pr, pe)


		#inner = self.inner_input.call(all_pairs)
		hidden = self.dense1.call(K.reshape(all_pairs, (self.product_count * self.person_count, 2 * self.style_width)))
		proj = self.dense2.call(hidden)
		proj = self.dense3.call(proj)
		proj = K.reshape(proj, (1, self.product_count, self.person_count))
		proj = K.tile(proj, [self.batch_size,1,1])

		# proj = K.dot(self.product, K.transpose(self.person))

		mul = proj * x
		# mul = mul * self.w1 + self.b1
		# mul = K.sigmoid(mul)
		# mul = mul * self.w2 + self.b2

		return mul

	def compute_output_shape(self, input_shape):
		return input_shape