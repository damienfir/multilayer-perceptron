import numpy as np
import matplotlib.pyplot as plt

results = 'results/'
plots = 'plots/'

def plot_mlp_errors(fname):
	data = np.loadtxt(results+fname+'.txt')
	plt.figure()
	plt.plot(data[0],data[1].T)
	plt.plot(data[0],data[2].T)
	plt.savefig(plots+fname+'_errors.png')
	plt.figure()
	plt.plot(data[0],data[3].T)
	plt.savefig(plots+fname+'_classerror.png')

plot_mlp_errors('mlp_binary')
# plot_mlp_errors('mlp_5class')
