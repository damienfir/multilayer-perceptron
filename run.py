import sys
import numpy as np, scipy.io
import mlp
import stream_utils as streams
import lstsq
import early_stopping

errors = classerrors = []
for i in range(0, 10):
	print 'Run', i + 1
	classifier = lstsq.LeastSquares(0.6)
	x_lefts, x_rights, ts = streams.training_5class().all()
	classifier.train(x_lefts, x_rights, ts)
	error, classerror = classifier.normalized_error(streams.testing_5class())
	errors, classerrors = errors + [error], classerrors + [classerror]

np.savetxt('results/lstsq_test_errors.txt', np.array(errors))
np.savetxt('results/lstsq_test_classerrors.txt', np.array(classerrors))
