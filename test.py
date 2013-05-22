import numpy as np
import datetime
import stream_utils as streams
import mlp

training, validation = streams.validation_binary()
classifier = mlp.MLP(10, 10, k=2, nu=0.0001, mu=0.1)

while True:
	x_left, x_right, t = training.all()
	grads = classifier.gradients(x_left, x_right, t)
	gradient = sum(map(lambda x: np.sum(np.power(x, 2.0)), grads))
	print "grad sum =",gradient
	classifier.train(x_left, x_right, t)
	error, _ = classifier.normalized_error(validation)
	print "error =",error

