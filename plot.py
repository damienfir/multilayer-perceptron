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
	plt.figure()
	labels = ['eta=2e-2 mu=5e-2 bs=5', 'eta=2e-2 mu=5e-2 bs=10', 'eta=2e-2 mu=5e-2 bs=20', 'eta=2e-2 mu=1.5e-1 bs=2', 'eta=2e-2 mu=2e-1 bs=20']
	colors = ['red', 'blue', 'green', 'black', 'pink']
	for i in range(0, 5):
		with open(results + 'logistic_' + str(i + 1) + '.txt', 'r') as csvfile:
			reader = csv.reader(csvfile, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
			lines = [line for line in reader]
#			max_length = max(map(lambda x: len(x), lines))
#			lines = map(lambda line: line + ((max_length - len(line)) * [line[-1]]), lines)
			min_length = min(map(lambda x: len(x), lines))
			lines = map(lambda line: line[:min_length], lines)
			data = np.array(lines, dtype=np.float)[:,6:]
			print np.mean(data[:,4:6], 0)
			nu, mu, bs = lines[0][0:3]
			plt.plot(np.mean(data, 0), label=labels[i], color=colors[i])
			plt.fill_between(range(0, data.shape[1]), np.max(data, 0), np.min(data, 0), facecolor=colors[i], alpha=0.2)
	plt.xlabel('Epoch number')
	plt.ylabel('Logistic error')
	plt.legend()
	plt.savefig('report/logistic_selection.pdf')
	print " +---> DONE"

# plot_lstsq_errors()
# plot_logistic_errors()

def plot_errors(data):
	plt.figure()
	plt.plot(data[0,:], label='Training set')
	plt.plot(data[2,:], label='Validation set')
	plt.xlabel('Epoch number')
	plt.ylabel('Logistic error')
	plt.legend()

def plot_errors_mlp2():
	data = np.loadtxt('plots/errors_mlp2.txt')
	plot_errors(data[:,:50])
	plt.savefig('report/errors_mlp2.pdf')

def plot_errors_mlp5():
	data = np.loadtxt('plots/errors_mlp5.txt')
	plot_errors(data)
	plt.savefig('report/errors_mlp5.pdf')
	plt.figure()
	plt.plot(data[1,:]*100, label='Training set')
	plt.plot(data[3,:]*100, label='Validation set')
	plt.xlabel('Epoch number')
	plt.ylabel('Classification error (%)')
	plt.legend()
	plt.savefig('report/classerrors_mlp5.pdf')

def plot_errors_logistic():
	data = np.loadtxt('plots/errors_logistic.txt')
	plot_errors(data)
	plt.savefig('report/errors_logistic.pdf')
	plt.figure()
	plt.plot(data[1,:]*100, label='Training set')
	plt.plot(data[3,:]*100, label='Validation set')
	plt.xlabel('Epoch number')
	plt.ylabel('Classification error (%)')
	plt.legend()
	plt.savefig('report/classerrors_logistic.pdf')

def plot_comparative_mlp2():
	data = np.loadtxt('plots/errors_comparative_mlp2.txt')
	plt.figure()
	plt.plot(data[0,:], label='eta=1e-3 mu=1e-1 H1=20 H2=50 bs=10')
	plt.plot(data[1,:], label='eta=1e-3 mu=1e-1 H1=1  H2=1  bs=10')
	plt.plot(data[2,:], label='eta=1e-5 mu=1e-1 H1=20 H2=50 bs=10')
	plt.plot(data[3,:], label='eta=1e-3 mu=5e-1 H1=20 H2=50 bs=10')
	plt.plot(data[4,:], label='eta=1e-3 mu=1e-1 H1=20 H2=50 bs=1')
	plt.xlabel('Epoch number')
	plt.ylabel('Logistic error')
	plt.legend(fontsize=14)
	plt.savefig('report/comparative_mlp2.pdf')

def plot_testing_errors():
	mlp2 = np.loadtxt('testing/mlp2_testing.txt')[:,1]
	mlp5 = np.loadtxt('testing/mlp5_testing.txt')[:,1]
	log = np.loadtxt('testing/logistic_testing.txt')[:,1]
	lstsq = np.loadtxt('testing/lstsq_test_classerrors.txt')
	plt.boxplot([mlp2,mlp5,log,lstsq])
	plt.ylabel('Misclassification rate')
	plt.xticks(np.arange(6), ('','MLP binary','MLP 5-class','Logistic','Squared',''))
	plt.savefig('report/testing_boxplot.pdf')
	print "mean, std mlp2:", mlp2.mean(), ', ', mlp2.std()
	print "mean, std mlp5:", mlp5.mean(), ', ', mlp5.std()
	print "mean, std log:", log.mean(), ', ', log.std()
	print "mean, std lstsq:", lstsq.mean(), ', ', lstsq.std()

def plot_errors_mlp5_overfitting():
	data = np.loadtxt('plots/errors_mlp5_overfitting.txt')
	plot_errors(data[:,:400])
	plt.savefig('report/errors_mlp5_overfitting.pdf')


def plot_confusion_matrix(labels,k):
	mat = np.zeros((k,k))
	for i in range(k):
		for j in range(k):
			mat[i,j] = np.sum((labels[0,:] == i+1) == (labels[1,:] == j+1))
	plt.matshow(mat)
	# cax = plt.matshow(mat)
	# plt.colorbar(cax)
	plt.xlabel('Estimated')
	plt.ylabel('Class')

def plot_confusion_matrices():
	plot_confusion_matrix(0.5 * (np.loadtxt('plots/confusion_mlp2.txt') + 1) + 1,2)
	plt.savefig('report/confusion_mlp2.pdf')
	plot_confusion_matrix(np.loadtxt('plots/confusion_mlp5_2.txt'),5)
	plt.savefig('report/confusion_mlp5.pdf')
	plot_confusion_matrix(np.loadtxt('plots/confusion_logistic.txt'),5)
	plt.savefig('report/confusion_logistic.pdf')
	plot_confusion_matrix(np.loadtxt('plots/confusion_lstsq.txt'),5)
	plt.savefig('report/confusion_lstsq.pdf')



def plot_all():
	#plot_errors('plots/errors_mlp2.txt')
	# plot_errors('plots/errors_mlp5.txt')
	#plot_errors('plots/errors_logistic.txt')
	# plot_errors_mlp2()
	# plot_errors_mlp5()
	# plot_errors_mlp5_overfitting()
	plot_comparative_mlp2()
	# plot_testing_errors()
	# plot_errors_lstsq()
	# plot_errors_logistic()
	# plot_confusion_matrices()







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


def generate_confusion_matrix(classifier,streams,testing,count=20):
	training,validation = streams
	x_left,x_right,t = testing.all()
	if t.shape[0] > 1:
		t = np.argmax(t,0)
	saved,errors,seconds = early_stopping.run(training,validation,classifier,max_time=300,count=count)
	t_ = saved.classify(x_left,x_right)
	return np.vstack((t,t_))

def generate_confusion_mlp2():
	classifier = mlp.MLP(20,50, nu=1e-3, mu=1e-1, k=2)
	confusion = generate_confusion_matrix(classifier,streams.validation_binary(),streams.testing_binary())
	np.savetxt('plots/confusion_mlp2.txt', confusion)

def generate_confusion_mlp5():
	classifier = mlp.MLP(60,10, nu=1e-3, mu=1e-1, k=5)
	confusion = generate_confusion_matrix(classifier,streams.validation_5class(),streams.testing_5class())
	np.savetxt('plots/confusion_mlp5_2.txt', confusion)

def generate_confusion_logistic():
	classifier = logistic.LogisticLoss(nu=2e-2,mu=5e-2)
	confusion = generate_confusion_matrix(classifier,streams.validation_5class(),streams.testing_5class(),count=5)
	np.savetxt('plots/confusion_logistic.txt', confusion)

def generate_confusion_lstsq():
	classifier = lstsq.LeastSquares(0.6)
	x_left,x_right,t = streams.training_5class().all()
	classifier.train(x_left, x_right, t)
	x_left,x_right,t = streams.testing_5class().all()
	t_ = classifier.classify(x_left, x_right)
	np.savetxt('plots/confustion_lstsq.txt', np.vstack((np.argmax(t,0),t_)))
	
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
	d = generate_errors(logistic.LogisticLoss(nu=2e-2,mu=5e-2), streams.validation_5class())
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
	# generate_confusion_mlp2()
	# generate_confusion_mlp5()
	generate_confusion_logistic()

# generate_confusion_lstsq()
# generate_all()
plot_all()
