import numpy as np

def sigmoid(x):
	return np.float(1.0) / (np.float(1.0) + np.exp(-x))

def dxsigmoid(x):
	s = sigmoid(x)
	return m(s, np.float(1.0) - s)

def tanh(x):
	return np.tanh(x)

def dxtanh(x):
	tmp = tanh(x)
	return np.float(1.0) - m(tmp, tmp)

def lsexp(v, direction=0):
	return np.log(np.sum(np.exp(v), direction))

def m(*args):
	args = list(args)
	tmp = args.pop()
	for i in args:
		tmp = np.multiply(tmp, i)
	return tmp
