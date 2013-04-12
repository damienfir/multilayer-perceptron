import scipy.io, numpy as np


def load_data():
	norb = scipy.io.loadmat('processed_binary.mat')
	
def load_param():
	param = scipy.io.loadmat('param_binary.mat')


def forward_pass(x_l, x_r, param):
	# first layer
	a1_left = param.w1_left.T * x_left + param.b1_left
	z1_left = tanh(a1_left)
	a1_right = param.w1_right.T * x_right + param.b1_right
	z1_right = tanh(a1_right)

	# second layer
	a2_left = param.w2_left.T * z1_left + param.b2_left
	a2_right = param.w2_right.T * z1_right + param.b2_right
	a2_rightleft = param.w2_rightleft.T * hstack(z1_left.T,z1_right.T) + param.b2_rightleft
	z2 = a2_rightleft * sigmoid(a2_left) * sigmoid(a2_right)

	# third layer
	a3 = param.w3.T * z2 + param.b3

	return a3


def sigmoid(x):
	return (1 + exp(-x))
