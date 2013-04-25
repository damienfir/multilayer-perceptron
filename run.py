import sys
import numpy as np
import streamer
import mlp, lstsq, logistic

path = 'norb/processed_binary.mat'
keys = ('training_left', 'training_right', 'training_cat')
stream = streamer.Stream(path, keys, count=1)
mlp_classifier = mlp.MLP(0.01, 0.1, 10, 10)
lstsq_classifier = lstsq.LeastSquares(0.1)
log_classifier = logistic.LogisticLoss(0.01, 0.1)

def classify(classifier):
	x_left, x_right, t = stream.next()
	result = classifier.classify(x_left, x_right)
	print "Result =?"
	print np.vstack([t, result])

def test_mlp():
	for _ in xrange(0, 100):
		x_left, x_right, t = stream.next()
		mlp_classifier.train(x_left, x_right, t)
	classify(mlp_classifier)

def test_least_squares():
	x_left, x_right, t = stream.all()
	lstsq_classifier.train(x_left, x_right, t)
	classify(lstsq_classifier)

def test_logistic():
	for _ in xrange(0, 100):
		x_left, x_right, t = stream.next()
		log_classifier.train(x_left, x_right, t)
	classify(log_classifier)

test_logistic()
