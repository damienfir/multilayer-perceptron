import utils
import random
import scipy.io
import numpy as np

class MLP:

	def __init__(self, gradient, H1, H2, dimension, seed=1234567890):
		self.gradient = gradient
		np.random.seed(seed)
		def rand(dim1, dim2):
			# We use dim1 + 1 here to model the b variable
			return (1.0 / dim1) * np.random.randn(dim1 + 1, dim2)
		self.w1_left = rand(dimension, H1)
		self.w1_right = rand(dimension, H1)
		self.w2_left = rand(H1, H2)
		self.w2_leftright = rand(2 * H1, H2)
		self.w2_right = rand(H1, H2)
		self.w3 = rand(H2, 1)
		self.dim = dimension
		self.H1 = H1
		self.H2 = H2
	
	def w_add(self, h):
		self.w1_left += h * np.ones([self.dim, self.H1])
		self.w1_right += h * np.ones([self.dim, self.H1])
		self.w2_left += h * np.ones([self.H1, self.H2])
		self.w2_leftright += h * np.ones([2*self.H1, self.H2])
		self.w2_right += h * np.ones([self.H1, self.H2])
		self.w3 += h
	
	def forward_pass(self, x_left, x_right):
		print x_left.shape, x_right.shape
		# first layer
		a1_left = self.w1_left.T * x_left
		z1_left = utils.tanh(a1_left)
		a1_right = self.w1_right.T * x_right
		z1_right = utils.tanh(a1_right)

		# second layer
		a2_left = self.w2_left.T * z1_left
		a2_right = self.w2_right.T * z1_right
		a2_leftright = self.w2_leftright.T * np.hstack(z1_left.T, z1_right.T)
		z2 = a2_leftright * utils.sigmoid(a2_left) * utils.sigmoid(a2_right)

		# third layer
		a3 = self.w3.T * z2

		return a1_left,a1_right,a2_left,a2_leftright,a2_right,a3

	def backward_pass(self, ass, t):
		a1_left,a1_right,a2_left,a2_leftright,a2_right,a3 = ass

		# third layer
		r3 = utils.sigmoid(a3) - 0.5 * (t + 1)
		g3 = r3 * z2

		# second layer
		r2_left = r3 * self.w3 * a2_leftright * utils.sigmoid(a2_left) * utils.sigmoid(-a2_left) * utils.sigmoid(a2_right)
		r2_right = r3 * self.w3 * a2_leftright * utils.sigmoid(a2_left) * utils.sigmoid(a2_right) * utils.sigmoid(-a2_right)
		r2_leftright = r3 * self.w3 * utils.sigmoid(a2_left) * utils.sigmoid(a2_right)
		g2_left = r2_left * z1_left
		g2_leftright = r2_leftright * np.hstack(z1_left, z1_right)
		g2_right = r2_right * z1_right
		
		# first layer
		r1_left = r2_left * np.hstack(self.w2_left, self.w2_leftright) * utils.dxtanh(np.hstack(a2_left, a2_leftright))
		r1_right = r2_right * np.hstack(self.w2_leftright, self.w2_right) * utils.dxtanh(np.hstack(a2_leftright, a2_right))
		g1_left = r1_left * x_left
		g1_right = r1_right * x_right

		return g1_left,g1_right,g2_left,g2_leftright,g2_right,g3

	def train(self, x_left, x_right, t):
		b = np.matrix(np.ones([1, x_left.shape[1]])) 
		print b.shape
		print x_left.shape
		x_left_b = np.append(x_left, (np.matrix(np.ones([1, x_left.shape[1]]))))
		x_right_b = np.append(x_right, (np.matrix(np.ones([1, x_right.shape[1]]))))
		self.ass = self.forward_pass(x_left_b, x_right_b)
		self.grads = self.backward_pass(ass, t)
		print self.ass
		print self.grads
		#self.gradient.descend(grads, 
