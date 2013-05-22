import datetime
import sys

import numpy as np
import matplotlib.pyplot as plt

import stream_utils as streams
import early_stopping
import mlp
import lstsq
import logistic


def plot_errors(fname):
	data = np.loadtxt(fname)
	plt.figure()
	plt.plot(data[0,:], label='training')
	plt.plot(data[2,:], label='validation')
	# plt.savefig(plots+fname+'_errors.png')
	plt.legend()
	plt.show()
	plt.figure()
	plt.plot(data[1,:], label='training')
	plt.plot(data[3,:], label='validation')
	# plt.savefig(plots+fname+'_classerror.png')
	plt.legend()
	plt.show()

def plot_errors_lstsq():
	pass

def plot_all():
	# plot_errors('plots/errors_mlp2.txt')
	plot_errors('plots/errors_mlp5.txt')
	# plot_errors('plots/errors_logistic.txt')


def generate_errors(classifier):
	max_time = 10
	out = np.zeros((4,0))
	start_time = datetime.datetime.now()
	training, validation = streams.validation_binary()
	stop = False
	while not stop:
		x_left, x_right, t = training.next(20)
		classifier.train(x_left, x_right, t)
		if training.looped:
			training_error, training_classerror = classifier.normalized_error(training)
			validation_error, validation_classerror = classifier.normalized_error(validation)
			
			d = np.transpose(np.array([[training_error, training_classerror, validation_error, validation_classerror]]))
			out = np.hstack((out, d))
			print d

			current_time = datetime.datetime.now()
			if (current_time - start_time).seconds > max_time:
				stop = True
	return out
	
def generate_errors_mlp2():
	d = generate_errors(mlp.MLP(10,10, nu=1e-4, mu=1e-1))
	np.savetxt('plots/errors_mlp2.txt', d)

def generate_errors_mlp5():
	d = generate_errors(mlp.MLP(60,10, nu=1e-3, mu=1e-1, k=5))
	np.savetxt('plots/errors_mlp5.txt', d)

def generate_errors_logistic():
	d = generate_errors(logistic.LogisticLoss(nu=1e-3,mu=1e-1))
	np.savetxt('plots/errors_logistic.txt', d)

def generate_errors_lstsq():
	avg_errors = np.zeros([1])
	for v in np.arange(1e-2,9*1e-2,1e-2):
		classifier = lstsq.LeastSquares(v)
		stream = test.stream_5class()
		x_left, x_right, t = stream.all()
		classifier.train(x_left, x_right, t)
		avg_error = test.classify_5class(classifier)
		avg_errors = np.append(avg_errors,avg_error)
		print avg_error
	print avg_errors

def generate_all():
	# generate_errors_mlp2()
	generate_errors_mlp5()
	# generate_errors_logistic()
	# generate_errors_lstsq()


# generate_all()
plot_all()
