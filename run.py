import sys
import numpy as np, scipy.io

import stream_utils as streams
import early_stopping

import mlp
import lstsq
import logistic


error = []

for i in range(10):
	classifier = mlp.MLP(20,50, nu=1e-3, mu=1e-1, k=2)
	training, validation = streams.validation_binary()
	testing = streams.testing_binary()
	trained, errors, seconds = early_stopping.run(training,validation,classifier,max_time=60)
	out = trained.normalized_error(testing)
	error.append(out)
	np.savetxt('mlp2_testing_1.txt', np.array(error))
