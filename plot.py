import datetime
import sys

import numpy as np
import matplotlib.pyplot as plt

import stream_utils as streams
import early_stopping
import mlp
import lstsq
import logistic


def plot_errors(data):
	plt.figure()
	plt.plot(data[0,:], label='Training set')
	plt.plot(data[2,:], label='Validation set')
	# plt.savefig(plots+fname+'_errors.png')
	plt.xlabel('Epoch number')
	plt.ylabel('Logistic error')
	plt.legend()
	# plt.show()
	# plt.figure()
	# plt.plot(data[1,:], label='training')
	# plt.plot(data[3,:], label='validation')
	# plt.savefig(plots+fname+'_classerror.png')
	# plt.legend()
	# plt.show()

def plot_comparative(data):
	plt.figure()
	plt.plot(data.T)
	plt.xlabel('Epoch number')
	plt.ylabel('Logistic error')
	plt.legend()
	plt.show()

def plot_errors_lstsq():
	pass

def plot_errors_logistic():
	pass

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
	plot_comparative(data)

def plot_errors_mlp5_overfitting():
	data = np.loadtxt('plots/errors_mlp5_overfitting.txt')
	plot_errors(data[:,:400])
	plt.savefig('plots/errors_mlp5_overfitting.pdf')


def plot_confusion_matrix(labels,k):
	print labels
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
	# plot_errors_mlp2()
	# plot_errors_mlp5()
	# plot_errors_mlp5_overfitting()
	# plot_comparative_mlp2()
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
	saved,errors,seconds = early_stopping.run(training,validation,classifier,max_time=2)
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
	d = generate_errors(mlp.MLP(20,50, nu=1e-3, mu=1e-1, k=2), streams.validation_binary())
	np.savetxt('plots/errors_mlp2.txt', d)

def generate_errors_mlp5():
	d = generate_errors(mlp.MLP(60,10, nu=1e-3, mu=1e-1, k=5), streams.validation_5class())
	np.savetxt('plots/errors_mlp5.txt', d)

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
	# generate_errors_mlp2()
	# generate_errors_mlp5()
	# generate_errors_mlp5_overfitting()
	# generate_comparative_mlp2()
	generate_confusion_mlp2()
	generate_confusion_mlp5()
	# generate_errors_logistic()
	# generate_errors_lstsq()


# generate_all()
plot_all()
