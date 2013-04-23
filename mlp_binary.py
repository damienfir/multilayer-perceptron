import utils
import random
import scipy.io
import numpy as np

def load_data():
	return scipy.io.loadmat('norb/processed_binary.mat')

H1,H2,nu,mu = 

def process():
	data = load_data()
	ws = initialize_ws(5, 5, 576)
	gradient = Gradient(nu, mu)
	while True:
		x_left,x_right = get_mini_batch_job()
		ass = forward_pass(x_left, x_right, ws)
		grads = backward_pass(ass, ts)
		ws = gradient.descend(grads, ws)

def initialize_ws(H1,H2,dimension):
	np.random.seed(123456)
	def rand(dim1, dim2):
		return (1.0 / dim1) * np.random.randn(dim1, dim2)
	w1_left = rand(dimension, H1)
	w1_right = rand(dimension, H1)
	w2_left = rand(H1, H2)
	w2_leftright = rand(2 * H1, H2)
	w2_right = rand(H1, H2)
	w3 = rand(H2, 1)
	return w1_left,w1_right,w2_left,w2_leftright,w2_right,w3

def load_param():
	param = scipy.io.loadmat('param_binary.mat')

def forward_pass(x_left, x_right, ws):
	w1_left,w1_right,w2_left,w2_leftright,w2_right,w3 = ws

	# first layer
	a1_left = w1_left.T * x_left
	z1_left = utils.tanh(a1_left)
	a1_right = w1_right.T * x_right
	z1_right = utils.tanh(a1_right)

	# second layer
	a2_left = w2_left.T * z1_left
	a2_right = w2_right.T * z1_right
	a2_leftright = w2_leftright.T * np.hstack(z1_left.T, z1_right.T)
	z2 = a2_leftright * utils.sigmoid(a2_left) * utils.sigmoid(a2_right)

	# third layer
	a3 = w3.T * z2

	return (a1_left,a1_right,a2_left,a2_leftright,a2_right,a3)

def backward_pass(a, t):
	a1_left,a1_right,a2_left,a2_leftright,a2_right,a3 = a

	# third layer
	r3 = utils.sigmoid(a.a3) - 0.5 * (t + 1)
	grad_w3 = r3 * z2

	# second layer
	r2_left = r3 * w3 * a2_leftright * utils.sigmoid(a2_left) * utils.sigmoid(-a2_left) * utils.sigmoid(a2_right)
	r2_right = r3 * w3 * a2_leftright * utils.sigmoid(a2_left) * utils.sigmoid(a2_right) * utils.sigmoid(-a2_right)
	r2_leftright = r3 * w3 * utils.sigmoid(a2_left) * utils.sigmoid(a2_right)
	grad_w2_left = r2_left * z1_left
	grad_w2_leftright = r2_leftright * np.hstack(z1_left, z1_right)
	grad_w2_right = r2_right * z1_right
	
	# first layer
	r1_left = r2_left * np.hstack(w2_left, w2_leftright) * utils.dxtanh(np.hstack(a2_left, a2_leftright))
	r1_right = r2_right * np.hstack(w2_leftright, w2_right) * utils.dxtanh(np.hstack(a2_leftright, a2_right))
	grad_w1_left = r1_left * x_left
	grad_w1_right = r1_right * x_right

	return (grad_w1_left,grad_w1_right,grad_w2_left,grad_w2_leftright,grad_w2_right,grad_w3)
