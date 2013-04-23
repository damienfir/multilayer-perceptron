import numpy as np

def sigmoid(x):
	return (1 + np.exp(-x))

def tanh(x):
	return np.tanh(x)

def dxtanh(x):
	tmp = tanh(x)
	return 1 - m(tmp, tmp)

def m(*args):
	tmp = list(args).pop()
	for i in args:
		tmp = np.multiply(tmp, i)
	return tmp
