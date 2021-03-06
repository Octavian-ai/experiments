
from keras.layers import Lambda
import keras.backend as K

# Take that keras
from tensorflow import float32

def assert_shape(tensor, shape, strict=False):
	if strict:
		assert hasattr(tensor, '_keras_shape'), f"{tensor.name} is missing _keras_shape"
	assert tensor.shape[1:] == shape, f"{tensor.name} is wrong shape, expected {shape} found {tensor.shape[1:]}"

def assert_mtx_shape(mtx, shape, name="matrix"):
	assert mtx.shape == shape, f"{name} is wrong shape, expected {shape} found {mtx.shape}"

def expand_dims(v, axis):
	return Lambda(lambda x: K.expand_dims(x,axis))(v)