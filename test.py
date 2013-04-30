import sys
import numpy as np, scipy.io
import mlp
import stream_utils as streams
from utils import *

classifier = mlp.MLP(10, 10, k=2)

training = streams.training_binary(count=10)
testing = streams.testing_binary()

for _ in xrange(0, 100):
	x_left, x_right, t = training.next()
	classifier.train(x_left, x_right, t)

errors = 0.0
for _ in xrange(0, testing.size):
	x_left, x_right, t = testing.next()
	result = classifier.classify(x_left, x_right)
	if t != result:
		errors += 1.0
print "Success rate =", 100 * (1.0 - errors / float(testing.size))

