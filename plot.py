import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

results = 'results/'
plots = 'plots/'

def plot_lstsq_errors():
	print "-- Plotting Least Squares Errors -------------------- :"
	data = np.loadtxt(results + 'lstsq_interval.txt')
	plt.figure()
	plt.boxplot(data[:,1:].T, whis=2.0)
	plt.xticks(range(1, len(data[:,0]) + 1), data[:,0], rotation=70)
	plt.savefig(plots + 'lstsq_interval.png')
	data = np.loadtxt(results + 'lstsq_errors.txt')
	plt.figure()
	plt.boxplot(data[:,1:].T, whis=2.0)
	plt.xticks(range(1, len(data[:,0]) + 1), data[:,0], rotation=70)
	plt.savefig(plots + 'lstsq_errors.png')
	data = np.loadtxt(results + 'lstsq_classerrors.txt')
	plt.figure()
	plt.boxplot(data[:,1:].T)
	plt.xticks(range(1, len(data[:,0]) + 1), data[:,0], rotation=70)
	plt.savefig(plots + 'lstsq_classerrors.png')
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
			plt.savefig(plots + 'logistic_descent_' + str(cnt) + '.png')
		plot_for_count('1')
		plot_for_count('2')
		plot_for_count('5')
		plot_for_count('10')
		plot_for_count('20')
		plot_for_count('50')
	print " +---> DONE"

plot_lstsq_errors()
plot_logistic_errors()

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
