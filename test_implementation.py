import sys
import scipy.io
import numpy as np
from utils import *
import stream_utils as streams
import mlp, lstsq, logistic

def test_mlp_binary():
	classifier = mlp.MLP(10, 10, k=2)
	training, validation = streams.validation_binary()
	for _ in xrange(0, 1000):
		x_left, x_right, t = training.next(count=10)
		classifier.train(x_left, x_right, t)
	error, classerror = classifier.normalized_error(validation)
	print "-- Binary MLP ------------------- :"
	print "   Average error                  :", error
	print "   Miss-classification percentage :", 100.0 * classerror

def test_mlp_5class():
	classifier = mlp.MLP(10, 10, k=5)
	training, validation = streams.validation_5class()
	for _ in xrange(0, 1000):
		x_left, x_right, t = training.next(count=10)
		classifier.train(x_left, x_right, t)
	error, classerror = classifier.normalized_error(validation)
	print "-- K-Way MLP -------------------- :"
	print "   Average error                  :", error
	print "   Miss-classification percentage :", 100.0 * classerror

def test_least_squares():
	classifier = lstsq.LeastSquares(0.1)
	training, validation = streams.validation_5class()
	x_left, x_right, t = training.all()
	classifier.train(x_left, x_right, t)
	x_left, x_right, t = validation.all()
	error, classerror = classifier.normalized_error(validation)
	print "-- Least Squares Regression ----- :"
	print "   Average error                  :", error
	print "   Miss-classification percentage :", 100.0 * classerror

def test_logistic():
	classifier = logistic.LogisticLoss(0.01, 0.1)
	training, validation = streams.validation_5class()
	for _ in xrange(0, 1000):
		x_left, x_right, t = training.next(count=10)
		classifier.train(x_left, x_right, t)
	x_left, x_right, t = validation.all()
	error, classerror = classifier.normalized_error(validation)
	print "-- Logistic Regression ---------- :"
	print "   Average error                  :", error
	print "   Miss-classification percentage :", 100.0 * classerror

test_mlp_binary()
test_mlp_5class()
test_least_squares()
test_logistic()

