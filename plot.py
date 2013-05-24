import csv
import sys
import datetime
import numpy as np
import matplotlib.pyplot as plt

import stream_utils as streams
import early_stopping
import mlp
import lstsq
import logistic

results = 'results/'
plots = 'plots/'

def plot_errors_lstsq(data):
	plt.figure()
	plt.boxplot(data[:,1:].T, whis=2.0)
	plt.xticks(range(1, len(data[:,0]) + 1), data[:,0], rotation=60)
	plt.ylabel('Squared error')
	plt.xlabel('Tikhonov regularizer')
	plt.gcf().subplots_adjust(bottom=0.15)
	plt.savefig(plots + 'lstsq_interval.pdf')
	data = np.loadtxt(results + 'lstsq_errors.txt')
	plt.figure()
	plt.boxplot(data[:,1:].T, whis=2.0)
	plt.xticks(range(1, len(data[:,0]) + 1), data[:,0], rotation=60)
	plt.ylabel('Squared error')
	plt.xlabel('Tikhonov regularizer')
	plt.gcf().subplots_adjust(bottom=0.15)
	plt.savefig(plots + 'lstsq_errors.pdf')
	data = np.loadtxt(results + 'lstsq_classerrors.txt')
	plt.figure()
	plt.boxplot(data[:,1:].T)
	plt.xticks(range(1, len(data[:,0]) + 1), data[:,0], rotation=60)
	plt.ylabel('Classification error rate')
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

# plot_lstsq_errors()
#plot_logistic_errors()

def plot_errors(data):
	plt.figure()
	plt.plot(data[0,:], label='Training set')
	plt.plot(data[2,:], label='Validation set')
	# plt.savefig(plots+fname+'_errors.png')
	plt.xlabel('Epoch number')
	plt.ylabel('Squared error')
	plt.legend()

def plot_errors_mlp2():
	data = np.loadtxt('plots/errors_mlp2.txt')
	plot_errors(data[:,:50])
	plt.savefig('plots/errors_mlp2.pdf')

def plot_errors_mlp5():
	data = np.loadtxt('plots/errors_mlp5.txt')
	plot_errors(data)
	plt.savefig('plots/errors_mlp5.pdf')
	plt.figure()
	plt.plot(data[1,:]*100, label='Training set')
	plt.plot(data[3,:]*100, label='Validation set')
	plt.xlabel('Epoch number')
	plt.ylabel('Classification error (%)')
	plt.legend()
	plt.savefig('plots/classerrors_mlp5.pdf')

def plot_comparative_mlp2():
	data = np.loadtxt('plots/errors_comparative_mlp2.txt')
	plt.figure()
	plt.plot(data[0,:], label='nu=1e-3 mu=1e-1 H1=20 H2=50 bs=10')
	plt.plot(data[1,:], label='nu=1e-3 mu=1e-1 H1=1  H2=1  bs=10')
	plt.plot(data[2,:], label='nu=1e-5 mu=1e-1 H1=20 H2=50 bs=10')
	plt.plot(data[3,:], label='nu=1e-3 mu=5e-1 H1=20 H2=50 bs=10')
	plt.plot(data[4,:], label='nu=1e-3 mu=1e-1 H1=20 H2=50 bs=1')
	plt.xlabel('Epoch number')
	plt.ylabel('Logistic error')
	plt.legend(fontsize=14)
	plt.savefig('plots/comparative_mlp2.pdf')

def plot_testing_errors():
	mlp2 = np.loadtxt('testing/mlp2_testing.txt')[:,1]
	mlp5 = np.loadtxt('testing/mlp5_testing.txt')[:,1]
	plt.boxplot([mlp2,mlp5])
	# plt.show()
	plt.ylabel('Misclassification rate')
	plt.xticks(np.arange(6), ('','MLP binary','MLP 5-class','Logistic','Linear',''))
	plt.savefig('plots/testing_boxplot.pdf')
	print "mean, std mlp2:", mlp2.mean(), ', ', mlp2.std()
	print "mean, std mlp5:", mlp5.mean(), ', ', mlp5.std()

def plot_errors_mlp5_overfitting():
	data = np.loadtxt('plots/errors_mlp5_overfitting.txt')
	plot_errors(data[:,:400])
	plt.savefig('plots/errors_mlp5_overfitting.pdf')


def plot_confusion_matrix(labels,k):
	mat = np.zeros((k,k))
	for i in range(k):
		for j in range(k):
			mat[i,j] = np.sum((labels[0,:] == i+1) == (labels[1,:] == j+1))
	plt.matshow(mat)
	plt.xlabel('Estimated')
	plt.ylabel('Class')

def plot_confusion_matrices():
	plot_confusion_matrix(0.5 * (np.loadtxt('plots/confusion_mlp2.txt') + 1) + 1,2)
	plt.savefig('plots/confusion_mlp2.pdf')
	plot_confusion_matrix(np.loadtxt('plots/confusion_mlp5.txt'),5)
	plt.savefig('plots/confusion_mlp5.pdf')
	# plot_confusion_matrix(np.loadtxt('plots/confusion_logistic.txt'),5)
	# plt.savefig('plots/confusion_logistic.pdf')
	# plot_confusion_matrix(np.loadtxt('plots/confusion_lstsq.txt'),5)
	# plt.savefig('plots/confusion_lstsq.pdf')



def plot_all():
	#plot_errors('plots/errors_mlp2.txt')
	# plot_errors('plots/errors_mlp5.txt')
	#plot_errors('plots/errors_logistic.txt')
	# plot_errors_mlp2()
	# plot_errors_mlp5()
	# plot_errors_mlp5_overfitting()
	# plot_comparative_mlp2()
	# plot_testing_errors()
	# plot_errors_lstsq()
	# plot_logistic_errors()
	plot_confusion_matrices()







def generate_errors(classifier, stream, max_time=300, count=10):
	out = np.zeros((4,0))
	start_time = datetime.datetime.now()
	training, validation = stream
	stop = False
	while not stop:
		x_left, x_right, t = training.next(count)
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


def generate_confusion_matrix(classifier,streams,testing):
	training,validation = streams
	x_left,x_right,t = testing.all()
	if t.shape[0] > 1:
		t = np.argmax(t,0)
	saved,errors,seconds = early_stopping.run(training,validation,classifier,max_time=100)
	t_ = saved.classify(x_left,x_right)
	return np.vstack((t,t_))

def generate_confusion_mlp2():
	classifier = mlp.MLP(20,50, nu=1e-3, mu=1e-1, k=2)
	confusion = generate_confusion_matrix(classifier,streams.validation_binary(),streams.testing_binary())
	np.savetxt('plots/confusion_mlp2.txt', confusion)

def generate_confusion_mlp5():
	classifier = mlp.MLP(60,10, nu=1e-3, mu=1e-1, k=5)
	confusion = generate_confusion_matrix(classifier,streams.validation_5class(),streams.testing_5class())
	np.savetxt('plots/confusion_mlp5.txt', confusion)

def generate_confusion_logistic():
	classifier = mlp.MLP(60,10, nu=1e-3, mu=1e-1, k=5)
	confusion = generate_confusion_matrix(classifier,streams.validation_5class(),streams.testing_5class())
	np.savetxt('plots/confusion_logistic.txt', confusion)

def generate_confusion_lstsq():
	classifier = mlp.MLP(60,10, nu=1e-3, mu=1e-1, k=5)
	confusion = generate_confusion_matrix(classifier,streams.validation_5class(),streams.testing_5class())
	np.savetxt('plots/confusion_lstsq.txt', confusion)
	
def generate_comparative_mlp2():
	t = 100
	d1 = generate_errors(mlp.MLP(20,50, nu=1e-3, mu=1e-1, k=2), streams.validation_binary(), t)
	d2 = generate_errors(mlp.MLP(1,1, nu=1e-3, mu=1e-1, k=2), streams.validation_binary(), t)
	d3 = generate_errors(mlp.MLP(20,50, nu=1e-5, mu=1e-1, k=2), streams.validation_binary(), t)
	d4 = generate_errors(mlp.MLP(20,50, nu=1e-3, mu=5e-1, k=2), streams.validation_binary(), t)
	d5 = generate_errors(mlp.MLP(20,50, nu=1e-5, mu=1e-1, k=2), streams.validation_binary(), t, 1)
	n = min(d1.shape[1],d2.shape[1],d3.shape[1],d4.shape[1],d5.shape[1])
	d = np.vstack((d1[2,:n],d2[2,:n],d3[2,:n],d4[2,:n],d5[2,:n]))
	np.savetxt('plots/errors_comparative_mlp2.txt', d)
	
def generate_errors_mlp2():
	d = generate_errors(mlp.MLP(100, 100, nu=1e-3, mu=1e-1, k=2), streams.validation_binary(training_ratio=0.05, count=100))
	np.savetxt('plots/errors_mlp2.txt', d)

def generate_errors_mlp5():
	d = generate_errors(mlp.MLP(100, 100, nu=1e-3, mu=1e-1, k=5), streams.validation_5class(training_ratio=0.5, count=100))
	np.savetxt('plots/errors_mlp2.txt', d)

def generate_errors_mlp5_overfitting():
	d = generate_errors(mlp.MLP(100,100, nu=1e-3, mu=1e-1, k=5), streams.validation_5class(training_ratio=0.5, count=100))
	np.savetxt('plots/errors_mlp5_overfitting.txt', d)

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
	#generate_errors_lstsq()
	# generate_errors_mlp2()
	# generate_errors_mlp5()
	# generate_errors_mlp5_overfitting()
	# generate_comparative_mlp2()
	generate_confusion_mlp2()
	generate_confusion_mlp5()
	# generate_errors_logistic()
	# generate_errors_lstsq()

#generate_all()
plot_all()
