import sys
import scipy.io
import numpy as np
from utils import *
import streamer
import mlp, lstsq, logistic

block_size = 10
mlp_classifier = mlp.MLP(0.01, 0.1, 10, 10)
lstsq_classifier = lstsq.LeastSquares(0.1)
log_classifier = logistic.LogisticLoss(0.01, 0.1)

def training_binary(count=block_size):
	path = 'norb/processed_binary.mat'
	keys = ('training_left', 'training_right', 'training_cat')
	return streamer.Stream(path, keys, count=count)

def validations_binary(count=1):
	path = 'norb/processed_binary.mat'
	keys = ('validation_left', 'validation_right', 'validation_cat')
	return streamer.Stream(path, keys, count=count)

def training_5class(count=block_size):
	path = 'norb/processed_5class.mat'
	keys = ('training_left', 'training_right', 'training_cat')
	return streamer.Stream(path, keys, count=count)

def validations_5class(count=1):
	path = 'norb/processed_5class.mat'
	keys = ('validation_left', 'validation_right', 'validation_cat')
	return streamer.Stream(path, keys, count=count)

def normalized_testing_stream(path, params_path, count):
	data = scipy.io.loadmat(params_path)
	mean_left, sigma_left = map(lambda x: np.tile(np.mat(x).T, (x.shape[0], count)), data['params_left'])
	mean_right, sigma_right = map(lambda x: np.tile(np.mat(x).T, (x.shape[0], count)), data['params_right'])
	def data_map(key, x):
		if key == 'test_left_s':    return m(1.0 / sigma_left, x - mean_left)
		elif key == 'test_right_s': return m(1.0 / sigma_right, x - mean_right)
		else:                       return x
	keys = ('test_left_s', 'test_right_s', 'test_cat_s')
	return streamer.Stream(path, keys, count=count, map=data_map)

def testing_binary(count=1):
	path = 'norb/norb_binary.mat'
	params_path = 'norb/processed_binary.mat'
	return normalized_testing_stream(path, params_path, count)

def testing_5class(count=1):
	path = 'norb/norb_5class.mat'
	params_path = 'norb/processed_5class.mat'
	return normalized_testing_stream(path, params_path, count)

def classify(classifier, validations, equals):
	errors = 0.0
	for _ in xrange(0, validations.size):
		x_left, x_right, t = validations.next()
		result = classifier.classify(x_left, x_right)
		# print t, result
		if not equals(t, result):
			errors += 1.0
	avg_error = errors / float(validations.size)
	# print "Success rate =", 100 * (1.0 - avg_error)
	return avg_error

def classify_binary(classifier):
	return classify(classifier, validations_binary(), lambda t,res: t - 2.0 == res) 

def classify_5class(classifier):
	return classify(classifier, validations_5class(), lambda t,res: t == res)

def test_mlp_binary():
	stream = training_binary()
	for _ in xrange(0, 100):
		x_left, x_right, t = stream.next()
		mlp_classifier.train(x_left, x_right, t)
	classify_binary(mlp_classifier)

def test_least_squares():
	stream = training_5class()
	x_left, x_right, t = stream.all()
	lstsq_classifier.train(x_left, x_right, t)
	classify_5class(lstsq_classifier)

def test_logistic():
	stream = training_5class()
	for _ in xrange(0, 100):
		x_left, x_right, t = stream.next()
		log_classifier.train(x_left, x_right, t)
	classify_5class(log_classifier)

test_mlp_binary()

