import csv
import datetime
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

import stream_utils as streams
import early_stopping
import mlp
import lstsq
import logistic

results = 'results/'
plots = 'plots/'

def plot_lstsq_errors():
	print "-- Plotting Least Squares Errors -------------------- :"
	data = np.loadtxt(results + 'lstsq_interval.txt')
	plt.figure()
	plt.boxplot(data[:,1:].T, whis=2.0)
	plt.xticks(range(1, len(data[:,0]) + 1), data[:,0], rotation=60)
	plt.ylabel('squared error')
	plt.xlabel('Tikhonov regularizer')
	plt.gcf().subplots_adjust(bottom=0.15)
	plt.savefig(plots + 'lstsq_interval.pdf')
	data = np.loadtxt(results + 'lstsq_errors.txt')
	plt.figure()
	plt.boxplot(data[:,1:].T, whis=2.0)
	plt.xticks(range(1, len(data[:,0]) + 1), data[:,0], rotation=60)
	plt.ylabel('squared error')
	plt.xlabel('Tikhonov regularizer')
	plt.gcf().subplots_adjust(bottom=0.15)
	plt.savefig(plots + 'lstsq_errors.pdf')
	data = np.loadtxt(results + 'lstsq_classerrors.txt')
	plt.figure()
	plt.boxplot(data[:,1:].T)
	plt.xticks(range(1, len(data[:,0]) + 1), data[:,0], rotation=60)
	plt.ylabel('classification error rate')
	plt.xlabel('Tikhonov regularizer')
	plt.gcf().subplots_adjust(bottom=0.15)
	plt.savefig(plots + 'lstsq_classerrors.pdf')
	print " +---> DONE"

def plot_logistic_errors():
	print "-- Plotting Logistic Regression Errors -------------- :"
	with open(results + 'logistic_descent.txt', 'r') as csvfile:
		reader = csv.reader(csvfile, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
		data = [(nu_repr, mu, count, error, classerror) for nu_repr, mu, count, l, s, error, classerror in reader]
		def plot_for_count(cnt):
			counts = [(nu_repr, mu, error, classerror) for nu_repr, mu, count, error, classerror in data if count == cnt]
			nus = list(np.reshape(np.array(map(lambda x: x[0], counts)), (12, 4))[:,0])
			nu_map = dict(zip(nus, range(len(nus))))
			X = np.reshape(np.array(map(lambda x: nu_map[x[0]], counts)), (12, 4))
			Y = np.reshape(np.array(map(lambda x: float(x[1]), counts)), (12, 4))
			Z = np.reshape(np.array(map(lambda x: float(x[2]), counts)), (12, 4))
			fig = plt.figure()
			ax = fig.gca(projection='3d')
			ax.set_xticks(range(len(nus)))
			ax.set_xticklabels(nus)
			ax.plot_wireframe(X, Y, Z, rstride=1, cstride=1)
			plt.savefig(plots + 'logistic_descent_' + str(cnt) + '.pdf')
		plot_for_count('1')
		plot_for_count('2')
		plot_for_count('5')
		plot_for_count('10')
		plot_for_count('20')
		plot_for_count('50')
	print " +---> DONE"

plot_lstsq_errors()
#plot_logistic_errors()

#def plot_mlp_errors(fname):
#	data = np.loadtxt(results + fname+'.txt')
#	plt.figure()
#	plt.plot(data[0],data[1].T)
#	plt.plot(data[0],data[2].T)
#	plt.savefig(plots+fname+'_errors.png')
#	plt.figure()
#	plt.plot(data[0],data[3].T)
#	plt.savefig(plots+fname+'_classerror.png')

#plot_mlp_errors('mlp_binary')
#plot_mlp_errors('mlp_5class')
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
	#plot_errors('plots/errors_mlp2.txt')
	plot_errors('plots/errors_mlp5.txt')
	#plot_errors('plots/errors_logistic.txt')


def generate_errors(classifier, streams):
	max_time = 10
	out = np.zeros((4,0))
	start_time = datetime.datetime.now()
	training, validation = streams
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
	d = generate_errors(mlp.MLP(100, 100, nu=1e-3, mu=1e-1, k=2), streams.validation_binary(training_ratio=0.05, count=100))
	np.savetxt('plots/errors_mlp2.txt', d)

def generate_errors_mlp5():
	d = generate_errors(mlp.MLP(100, 100, nu=1e-3, mu=1e-1, k=5), streams.validation_5class(training_ratio=0.5, count=100))
	np.savetxt('plots/errors_mlp5.txt', d)

def generate_errors_logistic():
	d = generate_errors(logistic.LogisticLoss(nu=1e-3,mu=1e-1), streams.validation_5class())
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
	#generate_errors_mlp2()
	generate_errors_mlp5()
	#generate_errors_logistic()
	#generate_errors_lstsq()


#generate_all()
#plot_all()
