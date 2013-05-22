import datetime
import sys

import numpy as np
import matplotlib.pyplot as plt

import stream_utils as streams
import early_stopping
import mlp
import lstsq
import logistic

results = 'results/'
plots = 'plots/'

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


def plot_all():
	pass


def generate_convergence():
	pass

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
	classifier = mlp.MLP(10,10, nu=1e-4, mu=1e-1)
	d = generate_errors(classifier)
	np.savetxt('plots/errors_mlp2.txt', d)

def generate_errors_logistic():
	classifier = logistic.LogisticLoss(nu=1e-3,mu=1e-1)
	d = generate_errors(classifier)
	np.savetxt('plots/errors_logistic.txt', d)

def generate_all():
	pass




# generate_errors_mlp2()
# generate_errors_logistic()

plot_errors('plots/errors_mlp2.txt')
plot_errors('plots/errors_logistic.txt')
