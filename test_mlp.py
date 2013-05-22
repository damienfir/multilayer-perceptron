import datetime
import stream_utils as streams
import numpy as np
import mlp

while True:
	stop = False
	saved = None
	seed = 293482934
	all_errors = last_errors = []
	start_time = datetime.datetime.now()
	training_stream, validation_stream = streams.validation_binary(seed=seed)
	classifier = mlp.MLP(50, 50, k=2, nu=3e-4, mu=1e-1)
	count = 10
	max_time = 100
	while not stop:
		x_left, x_right, t = training_stream.next(10)
		classifier.train(x_left, x_right, t)
		if training_stream.looped:
			error, classerror = classifier.normalized_error(validation_stream)
			all_errors = all_errors + [error]

			grad = classifier.gradients(x_left, x_right, t)
			print "gradients: ", np.sqrt(sum(map(lambda x: np.sum(np.array(x)**2), grad)))

			# save errors to check for function climbing
			if not last_errors or error < min(last_errors):
				last_errors = [error]
				saved = classifier.clone()
			elif error > last_errors[-1]:
				last_errors = last_errors + [error]
			else:
				last_errors = [min(last_errors), error]

			# if we notice we are climbing, stop training
			# if len(last_errors) > 3 or error == 0:
			# 	stop = True

			# if we have taken too long, stop training
			current_time = datetime.datetime.now()
			# print "errors:", last_errors
			# print "time:  ", (current_time - start_time).seconds
			# if (current_time - start_time).seconds > max_time:
			# 	stop = True

	current_time = datetime.datetime.now()
	seconds = (current_time - start_time).seconds
	# return saved, all_errors, seconds
