import datetime

def run(training_stream, validation_stream, classifier, count=10, max_time=6000):
	stop = False
	saved = None
	all_errors = last_errors = []
	start_time = datetime.datetime.now()
	while not stop:
		x_left, x_right, t = training_stream.next(count)
		classifier.train(x_left, x_right, t)
		if training_stream.looped:
			error, _ = classifier.error(validation_stream)
			all_errors = all_errors + [error]

			# save errors to check for function climbing
			if not last_errors or error < min(last_errors):
				last_errors = [error]
				saved = classifier.clone()
			elif error > last_errors[-1]:
				last_errors = last_errors + [error]
			else:
				last_errors = [min(last_errors), error]

			# if we notice we are climbing, stop training
			if len(last_errors) > 3 or error == 0:
				stop = True

			# if we have taken too long, stop training
			current_time = datetime.datetime.now()
			if (current_time - start_time).seconds > max_time:
				stop = True

	current_time = datetime.datetime.now()
	seconds = (current_time - start_time).seconds
	return saved, all_errors, seconds
