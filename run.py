import sys
import numpy as np, scipy.io
import mlp
import streamer
from utils import *

#classifier = mlp.MLP(100, 100, nu=lambda x: 0.01 / pow(float(x), 0.1))
classifier = mlp.MLP(100, 100, nu=0.01)

path = 'norb/processed_5class.mat'
keys = ('training_left', 'training_right', 'training_cat')
training = streamer.Stream(path, keys, count=10)

keys = ('validation_left', 'validation_right', 'validation_cat')
validation = streamer.Stream(path, keys)

def compute_error_rate(args):
	x_left, x_right, t = args
	result = classifier.classify(x_left, x_right)
	t = np.mat(t)
	errors = np.sum(np.not_equal(t, result), 1)
	#print result[np.not_equal(t, result)]
	#print t[np.not_equal(t, result)]
	return errors.flat[0], x_left.shape[1]

stop = False
while not stop:
	x_left, x_right, t = training.next()
	classifier.train(x_left, x_right, t)
	if training.looped:
		training_error = np.sum(classifier.error(*training.all()), 0)
		validation_error = np.sum(classifier.error(*validation.all()), 0)
		#training_error, training_size = compute_error_rate(training.all())
		#validation_error, validation_size = compute_error_rate(validation.all())
		print "Training error:", training_error, "out of", training.size,
		print "Validation error:", validation_error, "out of", validation.size
