import sys
import numpy as np, scipy.io
import mlp
import stream_utils as streams

classifier = mlp.MLP(100, 100, nu=0.01)
training = streams.training_5class()
validation = streams.validation_5class()

stop = False
while not stop:
	x_left, x_right, t = training.next()
	classifier.train(x_left, x_right, t)
	if training.looped:
		training_error, _ = classifier.error(*training.all())
		validation_error, _ = classifier.error(*validation.all())
		#training_error, training_size = compute_error_rate(training.all())
		#validation_error, validation_size = compute_error_rate(validation.all())
		print "Training error:", np.sum(training_error, 0), "out of", training.size,
		print "Validation error:", np.sum(validation_error, 0), "out of", validation.size
