from utils import *
import random
import scipy.io
import numpy as np

class MLP:

	def __init__(self, gradient, H1, H2, dimension, seed=1234567890):
		self.gradient = gradient
		self.dim = dimension
		self.H1 = H1
		self.H2 = H2
		np.random.seed(seed)
		def rand(dim1, dim2):
			# We use dim1 + 1 here to model the b variable
			return (1.0 / dim1) * np.matrix(np.random.randn(dim1 + 1, dim2))
		self.w1_left = rand(dimension, H1)
		self.w1_right = rand(dimension, H1)
		self.w2_left = rand(H1, H2)
		self.w2_leftright = rand(2 * H1, H2)
		self.w2_right = rand(H1, H2)
		self.w3 = rand(H2, 1)
	
	def w_add(self, h):
		self.w1_left += h * np.ones([self.dim, self.H1])
		self.w1_right += h * np.ones([self.dim, self.H1])
		self.w2_left += h * np.ones([self.H1, self.H2])
		self.w2_leftright += h * np.ones([2*self.H1, self.H2])
		self.w2_right += h * np.ones([self.H1, self.H2])
		self.w3 += h
	
	def forward_pass(self, x_left, x_right):
		# first layer
		b = np.matrix(np.ones([1, x_left.shape[1]])) 
		a1_left = self.w1_left.T * x_left
		z1_left = tanh(a1_left)
		z1_left_b = np.vstack([z1_left, b])
		a1_right = self.w1_right.T * x_right
		z1_right = tanh(a1_right)
		z1_right_b = np.vstack([z1_right, b])
		z1_leftright_b = np.vstack([z1_left, z1_right, b])

		# second layer
		a2_left = self.w2_left.T * z1_left_b
		a2_right = self.w2_right.T * z1_right_b
		a2_leftright = self.w2_leftright.T * z1_leftright_b
		z2 = m(a2_leftright, sigmoid(a2_left), sigmoid(a2_right))
		z2_b = np.vstack([z2, b])

		# third layer
		a3 = self.w3.T * z2_b

		zs = z1_left_b,z1_right_b,z2_b
		ass = a1_left,a1_right,a2_left,a2_leftright,a2_right,a3
		return zs, ass

	def backward_pass(self, zs, ass, t):
		z1_left,z1_right,z2 = zs
		a1_left,a1_right,a2_left,a2_leftright,a2_right,a3 = ass

		# third layer
		r3 = (sigmoid(a3) - 0.5 * (t + 1))
		g3 = r3 * z2.T

		# second layer
		print r3.shape, self.w3.shape, m(a2_leftright, sigmoid(a2_left), sigmoid(-a2_left), sigmoid(a2_right)).shape
		r2_left = r3 * (self.w3.T * m(a2_leftright, sigmoid(a2_left), sigmoid(-a2_left), sigmoid(a2_right))).T
		r2_right = r3 * (self.w3.T * m(a2_leftright, sigmoid(a2_left), sigmoid(a2_right), sigmoid(-a2_right))).T
		r2_leftright = (r3 * self.w3.T).T * m(sigmoid(a2_left), sigmoid(a2_right)).T
		g2_left = r2_left.T * z1_left
		g2_leftright = r2_leftright.T * np.hstack([z1_left, z1_right])
		g2_right = r2_right.T * z1_right
		
		# first layer
		print r2_leftright.shape
		print self.w2_left.shape
		w2_left = np.vstack([self.w2_left, self.w2_leftright[:self.H1,:], r2_leftright])
		w2_right = np.vstack([self.w2_leftright[self.H1:-1,:], self.w2_right, r2_leftright])
		print r2_left.shape, w2_left.shape
		print dxtanh(np.hstack([a1_left, a1_left])).shape
		r1_left = r2_left * w2_left.T * dxtanh(np.hstack([a1_left, a1_left]))
		r1_right = r2_right * w2_right * dxtanh(np.hstack([a1_right, a1_right]))
		g1_left = r1_left * x_left
		g1_right = r1_right * x_right

		return g1_left,g1_right,g2_left,g2_leftright,g2_right,g3

	def train(self, x_left, x_right, t):
		x_left = np.matrix(x_left).T
		x_right = np.matrix(x_right).T
		b = np.matrix(np.ones([1, x_left.shape[1]])) 
		x_left_b = np.vstack([x_left, b])
		x_right_b = np.vstack([x_right, b])
		zs, ass = self.forward_pass(x_left_b, x_right_b)
		grads = self.backward_pass(zs, ass, t)
		self.ass = ass
		self.grads = grads
		print ass
		print grads
		#self.gradient.descend(grads, 
