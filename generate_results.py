import numpy as np
import stream_utils as stream
import lstsq
import mlp
import logistic


def lstsq_errors():
	avg_errors = np.zeros([1])
	training = stream.training_5class()
	validation = stream.validation_5class()
	training_errors = validation_errors = classification_errors = np.zeros([0])
	v_range = np.arange(1e-2,9*1e-2,1e-2)
	for v in v_range:
		lstsq_classifier = lstsq.LeastSquares(v)
		lstsq_classifier.train(*training.all())
		error,_ = lstsq_classifier.error(*training.all())
		training_errors = np.append(training_errors, error)
		error, class_errors = lstsq_classifier.error(*validation.all())
		validation_errors = np.append(validation_errors, error)
		classification_errors = np.append(classification_errors, class_errors)
	store = np.vstack([v_range, training_errors, validation_errors, classification_errors])
	np.savetxt('results/lstsq.txt', store)


# MLP
def mlp_test(classifier, training, validation):
	training_errors = validation_errors = classification_errors = npoints = np.zeros([0])
	for _ in xrange(0, 100):
		x_left, x_right, t = training.next()
		classifier.train(x_left, x_right, t)
		npoints = np.append(npoints, training.count)
		errors, _ = classifier.error(*training.all())
		training_errors = np.append(training_errors, errors)
		errors, class_errors = classifier.error(*validation.all())
		validation_errors = np.append(validation_errors, errors)
		classification_errors = np.append(classification_errors, class_errors)
	return np.vstack([np.cumsum(npoints), training_errors, validation_errors, classification_errors])


def mlp_binary_errors():
	mlp_classifier = mlp.MLP(10, 10, 0.01, 0.1, k=2)
	training = stream.training_binary()
	validation = stream.validations_binary()
	store = mlp_test(mlp_classifier, training, validation)
	np.savetxt('results/mlp_binary.txt', store)


def mlp_5class_errors():
	mlp_classifier = mlp.MLP(10, 10, 0.01, 0.1, k=5)
	training = stream.training_5class()
	validation = stream.validation_5class()
	store = mlp_test(mlp_classifier, training, validation)
	np.savetxt('results/mlp_5class.txt', store)


def mlp_binary_comparative():
	pass

# mlp_binary_errors()
mlp_5class_errors()
# lstsq_errors()
