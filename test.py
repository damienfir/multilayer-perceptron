import sys
import numpy as np
import streamer
import mlp, lstsq, logistic

block_size = 10
mlp_classifier = mlp.MLP(0.01, 0.1, 10, 10)
lstsq_classifier = lstsq.LeastSquares(0.1)
log_classifier = logistic.LogisticLoss(0.01, 0.1)

def stream_binary():
	path = 'norb/processed_binary.mat'
	keys = ('training_left', 'training_right', 'training_cat')
	return streamer.Stream(path, keys, count=block_size)

def validations_binary():
	path = 'norb/processed_binary.mat'
	keys = ('validation_left', 'validation_right', 'validation_cat')
	return streamer.Stream(path, keys, count=block_size)

def stream_5class():
	path = 'norb/processed_5class.mat'
	keys = ('training_left', 'training_right', 'training_cat')
	return streamer.Stream(path, keys, count=block_size)

def validations_5class():
	path = 'norb/processed_5class.mat'
	keys = ('validation_left', 'validation_right', 'validation_cat')
	return streamer.Stream(path, keys, count=block_size)

def classify(classifier, validations):
	errors = 0.0
	for _ in xrange(0, validations.size):
		x_left, x_right, t = validations.next()
		result = classifier.classify(x_left, x_right)
		if result != t:
			errors += 1.0
	print "Success rate =", errors / float(validations.size)

def classify_binary(classifier):
	classify(classifier, validations_binary())

def classify_5class(classifier):
	classify(classifier, validations_5class())

def test_mlp_binary():
	stream = stream_binary()
	for _ in xrange(0, 100):
		x_left, x_right, t = stream.next()
		mlp_classifier.train(x_left, x_right, t)
	classify_binary(mlp_classifier)

def test_least_squares():
	stream = stream_5class()
	x_left, x_right, t = stream.all()
	lstsq_classifier.train(x_left, x_right, t)
	classify_5class(lstsq_classifier)

def test_logistic():
	stream = stream_5class()
	for _ in xrange(0, 100):
		x_left, x_right, t = stream.next()
		log_classifier.train(x_left, x_right, t)
	classify_5class(log_classifier)

test_mlp_binary()
