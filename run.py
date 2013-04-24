import sys
import numpy as np
import streamer
import mlp, lstsq, logistic

path = 'norb/processed_binary.mat'
keys = ('training_left', 'training_right', 'training_cat')
stream = streamer.Stream(path, keys, count=10)
mlp_classifier = mlp.MLP(0.01, 0.1, 1, 1)
lstsq_classifier = lstsq.LeastSquares(0.1)
log_classifier = logistic.LogisticLoss(0.01, 0.1)

def verify_mlp_gradients(inputs, count):
	sys.stdout.write("Verifying gradient computation ... ")
	sys.stdout.flush()
	all_errors = []
	for _ in xrange(0, inputs):
		x_left, x_right, t = stream.next()
		errors = mlp_classifier.verify(x_left, x_right, t, count=count)
		all_errors.extend(errors)
	print "[DONE]"
	if all_errors:
		print "Errors found:"
		for idx, x, y, mlp_result, result in all_errors:
			print "ws[" + str(idx) + "][" + str(x) + "," + str(y) + "] ->",
			print "mlp gradient of " + str(mlp_result) + " vs directional of " + str(result)
		print "Total:", len(all_errors)
	else:
		print "No errors uncovered after " + str(count) + " random directions in " + str(inputs) + " inputs"

def train_mlp():
	for _ in xrange(0, 100):
		x_left, x_right, t = stream.next()
		mlp_classifier.train(x_left, x_right, t)

def train_least_squares():
	x_left, x_right, t = stream.all()
	lstsq_classifier.train(x_left, x_right, t)

def train_logistic():
	for _ in xrange(0, 100):
		x_left, x_right, t = stream.next()
		log_classifier.train(x_left, x_right, t)


verify_mlp_gradients(50, 100)
# train_logistic()
# x_left, x_right, t = stream.next()
# result = log_classifier.classify(x_left, x_right)
# print "Result =?"
# print np.vstack([t, result])
