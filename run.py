import sys
import numpy as np, scipy.io
import mlp
import stream_utils as streams
import lstsq
import early_stopping
import logistic


error = []

for i in range(10):
	# classifier = mlp.MLP(60,10, nu=1e-3, mu=1e-1, k=5)
	classifier = logistic.LogisticLoss(nu=1e-3, mu=1e-1)
	training, validation = streams.validation_5class()
	testing = streams.testing_5class()
	trained, errors, seconds = early_stopping.run(training,validation,classifier,max_time=60)
	out = trained.normalized_error(testing)
	error.append(out)
	np.savetxt('logistic_testing.txt', np.array(error))
