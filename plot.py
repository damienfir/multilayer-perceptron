import numpy as np
import matplotlib.pyplot as plt

results = 'results/'

def plot_mlp_binary():
	data = np.loadtxt(results+'mlp_binary.txt')
	plt.plot(data[0],data[1:4].T)
	plt.show()


plot_mlp_binary()
