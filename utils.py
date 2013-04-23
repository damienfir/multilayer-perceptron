import numpy as np

def sigmoid(x):
	return 1.0 / (1.0 + np.exp(-x))

def dxsigmoid(x):
	s = sigmoid(x)
	return m(s, 1.0 - s)

def tanh(x):
	return np.tanh(x)

def dxtanh(x):
	tmp = tanh(x)
	return 1.0 - m(tmp, tmp)

def m(*args):
	tmp = list(args).pop()
	for i in args:
		tmp = np.multiply(tmp, i)
	return tmp
