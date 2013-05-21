import scipy.io
import numpy as np
from utils import *
import streamer

_keys = ('x_left', 'x_right', 't')

_path_binary = 'norb/processed_binary.mat'
def _data_map_binary(key, x):
	if key == _keys[0] or key == _keys[1]: return np.mat(x, dtype=np.float)
	elif key == _keys[2]:                  return np.mat(x, dtype=np.float) - 2.0

_path_5class = 'norb/processed_5class.mat'
def _data_map_5class(key, x):
	if key == _keys[0] or key == _keys[1]: return np.mat(x, dtype=np.float)
	elif key == _keys[2]:                  return np.mat(np.vstack(map(lambda k: x == k, range(0, 5))), dtype=np.float)

def _validation_split(path, keys, data_map, training_ratio, count, seed):
	all_data = scipy.io.loadmat(path)
	data = map(lambda key: data_map(key, all_data[key]), keys)
	size = data[0].shape[1]
	indices = range(0, size)
	random = np.random.RandomState(seed)
	random.shuffle(indices)
	if count != None:
		indices = indices[:count]
	training_count = int(training_ratio * float(len(indices)))
	training_indices = indices[:training_count]
	validation_indices = indices[training_count:]
	training_stream = streamer.Stream(data, indices=training_indices)
	validation_stream = streamer.Stream(data, indices=validation_indices)
	return training_stream, validation_stream

def validation_binary(training_ratio=0.66, count=None, seed=None):
	return _validation_split(_path_binary, _keys, _data_map_binary, training_ratio, count, seed)

def training_binary(count=None, seed=None):
	training_stream, validation_stream = validation_binary(training_ratio=1.0, count=count, seed=seed)
	assert(validation_stream.size == 0)
	return training_stream

def validation_5class(training_ratio=0.66, count=None, seed=None):
	return _validation_split(_path_5class, _keys, _data_map_5class, training_ratio, count, seed)

def training_5class(count=None, seed=None):
	training_stream, validation_stream = validation_5class(training_ratio=1.0, count=count, seed=seed)
	assert(validation_stream.size == 0)
	return training_stream

def _cross_validation(path, keys, data_map, splits, count, seed):
	all_data = scipy.io.loadmat(path)
	data = map(lambda key: data_map(key, all_data[key]), keys)
	size = data[0].shape[1]
	indices = range(0, size)
	random = np.random.RandomState(seed)
	random.shuffle(indices)
	if count != None:
		indices = indices[:count]
	validation_count = int(float(len(indices)) / float(splits))
	for i in range(0, splits):
		start, end = i * validation_count, (i + 1) * validation_count
		training_indices = indices[:start] + indices[end:]
		validation_indices = indices[start:end]
		training_stream = streamer.Stream(data, indices=training_indices)
		validation_stream = streamer.Stream(data, indices=validation_indices)
		yield training_stream, validation_stream

def cross_validation_binary(splits=10, count=None, seed=None):
	return _cross_validation(_path_binary, _keys, _data_map_binary, splits, count, seed)

def cross_validation_5class(splits=10, count=None, seed=None):
	return _cross_validation(_path_5class, _keys, _data_map_5class, splits, count, seed)

def testing_binary(count=None, seed=None):
	path = 'norb/norb_binary.mat'
	keys = ('test_left_s', 'test_right_s', 'test_cat_s')
	all_data = scipy.io.loadmat(path)
	size = all_data[keys[0]].shape[1]
	indices = range(0, size)
	random = np.random.RandomState(seed)
	random.shuffle(indices)
	if count != None:
		indices = indices[:count]

	params_path = 'norb/processed_binary.mat'
	params = scipy.io.loadmat(params_path)
	mean_left, sigma_left = map(lambda x: np.tile(np.mat(x).T, (1, len(indices))), params['params_left'])
	mean_right, sigma_right = map(lambda x: np.tile(np.mat(x).T, (1, len(indices))), params['params_right'])
	def data_map(key, x):
		if   key == keys[0]: return m(1.0 / sigma_left, np.mat(x, dtype=np.float) - mean_left)
		elif key == keys[1]: return m(1.0 / sigma_right, np.mat(x, dtype=np.float) - mean_right)
		elif key == keys[2]: return np.mat(x, dtype=np.float) - 2.0

	data = map(lambda key: data_map(key, all_data[key]), keys)
	return streamer.Stream(data, indices=indices)

def testing_5class(count=1):
	path = 'norb/norb_5class.mat'
	keys = ('test_left_s', 'test_right_s', 'test_cat_s')
	all_data = scipy.io.loadmat(path)
	size = all_data[keys[0]].shape[1]
	indices = range(0, size)
	random = np.random.RandomState(seed)
	random.shuffle(indices)
	if count != None:
		indices = indices[:count]

	params_path = 'norb/processed_5class.mat'
	params = scipy.io.loadmat(params_path)
	mean_left, sigma_left = map(lambda x: np.tile(np.mat(x).T, (1, len(indices))), params['params_left'])
	mean_right, sigma_right = map(lambda x: np.tile(np.mat(x).T, (1, len(indices))), params['params_right'])
	def data_map(key, x):
		if   key == keys[0]: return m(1.0 / sigma_left, np.mat(x, dtype=np.float) - mean_left)
		elif key == keys[1]: return m(1.0 / sigma_right, np.mat(x, dtype=np.float) - mean_right)
		elif key == keys[2]: return np.mat(np.vstack(map(lambda k: x == k, range(0, 5))), dtype=np.float)

	data = map(lambda key: data_map(key, all_data[key]), keys)
	return streamer.Stream(data, indices=indices)
