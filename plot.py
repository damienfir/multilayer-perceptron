import numpy as np
import matplotlib.pyplot as plt

results = 'results/'

def plot_mlp_binary():
	data = np.loadtxt(results+'mlp_binary.txt')
	plt.figure()
	plt.plot(data[0],data[1].T)
	plt.plot(data[0],data[2].T)
	plt.show()
	plt.figure()
	plt.plot(data[0],data[3].T)
	plt.show()

plot_mlp_binary()
