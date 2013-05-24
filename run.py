import sys
import numpy as np, scipy.io

import stream_utils as streams
import early_stopping

import mlp
import lstsq
import logistic


error = []

for i in range(10):
	classifier = logistic.LogisticLoss(nu=2e-2,mu=5e-2)
	training, validation = streams.validation_5class()
	testing = streams.testing_5class()
	trained, errors, seconds = early_stopping.run(training,validation,classifier,max_time=60,count=5)
	out = trained.normalized_error(testing)
	error.append(out)
	np.savetxt('testing/logistic_testing.txt', np.array(error))
