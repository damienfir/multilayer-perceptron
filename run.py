import sys
import numpy as np, scipy.io

import stream_utils as streams
import early_stopping

import mlp
import lstsq
import logistic


error = []

for i in range(1):
	# classifier = logistic.LogisticLoss(nu=2e-2,mu=5e-2)
	classifier = mlp.MLP(60, 10, nu=1e-3, mu=1e-1, k=5)
	training, validation = streams.validation_5class()
	testing = streams.testing_5class()
	trained, errors, seconds = early_stopping.run(training,validation,classifier,max_time=1,count=5)
	x_left, x_right, t = testing.all()
	trained.tiyi(x_left, x_right, t)
	# out = trained.normalized_error(testing)
	# error.append(out)
	# np.savetxt('testing/mlp5_tiyi.txt', np.array(error))
