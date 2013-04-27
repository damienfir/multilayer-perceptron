import sys
import numpy as np, scipy.io
import mlp
import streamer
from utils import *

classifier = mlp.MLP(0.01, 0.1, 10, 10)

path = 'norb/processed_binary.mat'
keys = ('training_left', 'training_right', 'training_cat')
training = streamer.Stream(path, keys, count=10)

path = 'norb/norb_binary.mat'
keys = ('test_left_s', 'test_right_s', 'test_cat_s')
testing = streamer.Stream(path, keys)

mat = scipy.io.loadmat('norb/processed_binary.mat')
mean_left, sigma_left = map(lambda x: np.mat(x).T, mat['params_left'])
mean_right, sigma_right = map(lambda x: np.mat(x).T, mat['params_right'])

for _ in xrange(0, 100):
	x_left, x_right, t = training.next()
	classifier.train(x_left, x_right, t)

errors = 0.0
for _ in xrange(0, testing.size):
	x_left, x_right, t = testing.next()
	x_left = m(1.0 / sigma_left, x_left - mean_left)
	x_right = m(1.0 / sigma_right, x_right - mean_right)
	result = classifier.classify(x_left, x_right)
	if float(t) - 2 != result:
		errors += 1.0
print "Success rate =", 100 * (1.0 - errors / float(testing.size))

