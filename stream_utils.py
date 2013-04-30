import scipy.io
import numpy as np
from utils import *
import streamer

def binary_stream(path, keys, count):
	assert len(keys) == 3
	def data_map(key, x):
		if key == keys[0] or key == keys[1]: return np.mat(x, dtype=np.float)
		elif key == keys[2]:                 return np.mat(x, dtype=np.float) - 2.0
	return streamer.Stream(path, keys, count=count, map=data_map)

def training_binary(count=10):
	path = 'norb/processed_binary.mat'
	keys = ('training_left', 'training_right', 'training_cat')
	return binary_stream(path, keys, count)

def validation_binary(count=1):
	path = 'norb/processed_binary.mat'
	keys = ('validation_left', 'validation_right', 'validation_cat')
	return binary_stream(path, keys, count)

def kclass_stream(path, keys, count):
	assert len(keys) == 3
	def data_map(key, x):
		if key == keys[0] or key == keys[1]: return np.mat(x, dtype=np.float)
		elif key == keys[-1]:                return np.mat(np.vstack(map(lambda k: x == k, range(0, 5))), dtype=np.float)
	return streamer.Stream(path, keys, count=count, map=data_map)

def training_5class(count=10):
	path = 'norb/processed_5class.mat'
	keys = ('training_left', 'training_right', 'training_cat')
	return kclass_stream(path, keys, count)

def validation_5class(count=1):
	path = 'norb/processed_5class.mat'
	keys = ('validation_left', 'validation_right', 'validation_cat')
	return kclass_stream(path, keys, count)

def testing_binary(count=1):
	path = 'norb/norb_binary.mat'
	params_path = 'norb/processed_binary.mat'
	keys = ('test_left_s', 'test_right_s', 'test_cat_s')
	data = scipy.io.loadmat(params_path)
	mean_left, sigma_left = map(lambda x: np.tile(np.mat(x).T, (1, count)), data['params_left'])
	mean_right, sigma_right = map(lambda x: np.tile(np.mat(x).T, (1, count)), data['params_right'])
	def data_map(key, x):
		if key == keys[0]:   return m(1.0 / sigma_left, np.mat(x, dtype=np.float) - mean_left)
		elif key == keys[1]: return m(1.0 / sigma_right, np.mat(x, dtype=np.float) - mean_right)
		elif key == keys[2]: return np.mat(x, dtype=np.float) - 2.0
	return streamer.Stream(path, keys, count=count, map=data_map)

def testing_5class(count=1):
	path = 'norb/norb_5class.mat'
	params_path = 'norb/processed_5class.mat'
	keys = ('test_left_s', 'test_right_s', 'test_cat_s')
	data = scipy.io.loadmat(params_path)
	mean_left, sigma_left = map(lambda x: np.tile(np.mat(x).T, (1, count)), data['params_left'])
	mean_right, sigma_right = map(lambda x: np.tile(np.mat(x).T, (1, count)), data['params_right'])
	def data_map(key, x):
		if key == keys[0]:   return m(1.0 / sigma_left, x - mean_left)
		elif key == keys[1]: return m(1.0 / sigma_right, x - mean_right)
		elif key == keys[2]: return np.mat(np.vstack(map(lambda k: x == k, range(0, 5))), dtype=np.float)
	return streamer.Stream(path, keys, count=count, map=data_map)
