import numpy as np
import test_implementation as test
import lstsq
import mlp


def lstsq_error():
	avg_errors = np.zeros([1])
	for v in np.arange(1e-2,9*1e-2,1e-2):
		classifier = lstsq.LeastSquares(v)
		stream = test.training_5class()
		x_left, x_right, t = stream.all()
		classifier.train(x_left, x_right, t)
		avg_error = test.classify_5class(classifier)
		avg_errors = np.append(avg_errors,avg_error)
		print avg_error
	print avg_errors


def mlp_binary_error():
	mlp_classifier = mlp.MLP(10, 10, 0.01, 0.1, k=2)
	training = test.training_binary()
	validations = test.validations_binary()
	training_errors = validation_errors = classification_errors = npoints = np.zeros([0])
	for _ in xrange(0, 100):
		x_left, x_right, t = training.next()
		mlp_classifier.train(x_left, x_right, t)
		# keep results
		npoints = np.append(npoints, training.count)
		errors, _ = mlp_classifier.error(*training.all())
		training_errors = np.append(training_errors, errors)
		errors, class_errors = mlp_classifier.error(*validations.all())
		validation_errors = np.append(validation_errors, errors)
		classification_errors = np.append(classification_errors, class_errors)
	npoints = np.cumsum(npoints)
	store = np.vstack([npoints, training_errors, validation_errors, classification_errors])
	np.savetxt('results/mlp_binary.txt',store)

mlp_binary_error()
