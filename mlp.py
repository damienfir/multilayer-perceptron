from utils import *
import gradient
import scipy.io
import numpy as np

class MLP:

	def __init__(self, H1, H2, nu=0.01, mu=0.1, k=5, dimension=576, seed=1234567890, copy=None):
		self.gradient = gradient.Gradient(nu, mu)
		self.H1 = H1
		self.H2 = H2
		self.dim = dimension
		self.k = k
		self.binary = k == 2
		if k < 2:
			raise Exception(str(k) + "-way classification makes no sense")
		if copy == None:
			random = np.random.RandomState(seed)
			def rand(dim1, dim2):
				# We use dim2 + 1 here to model the b variable
				return (1.0 / dim1) * np.mat(random.randn(dim1, dim2 + 1), dtype=np.float)
			self.w1_left = rand(H1, dimension)
			self.w1_right = rand(H1, dimension)
			self.w2_left = rand(H2, H1)
			self.w2_leftright = rand(H2, 2 * H1)
			self.w2_right = rand(H2, H1)
			self.w3 = rand(1, H2) if self.binary else rand(k, H2)
		else:
			self.w1_left = np.copy(copy.w1_left)
			self.w1_right = np.copy(copy.w1_right)
			self.w2_left = np.copy(copy.w2_left)
			self.w2_leftright = np.copy(copy.w2_leftright)
			self.w2_right = np.copy(copy.w2_right)
			self.w3 = np.copy(copy.w3)

	def clone(self):
		return MLP(self.H1, self.H2, nu=self.gradient.nu, mu=self.gradient.mu,
			k=self.k, dimension=self.dim, copy=self)

	def gradients(self, x_left, x_right, t):
		b = np.mat(np.ones([1, np.mat(x_left).shape[1]])) 
		x_left = np.mat(np.vstack([np.mat(x_left), b]), dtype=np.float)
		x_right = np.mat(np.vstack([np.mat(x_right), b]), dtype=np.float)
		zs, ass = self.forward_pass(x_left, x_right)
		grads = self.backward_pass(zs, ass, x_left, x_right, t)
		return grads

	@property
	def ws(self):
		return self.w1_left,self.w1_right,self.w2_left,self.w2_leftright,self.w2_right,self.w3

	def update_ws(self, ws):
		w1_left,w1_right,w2_left,w2_leftright,w2_right,w3 = ws
		self.w1_left = w1_left
		self.w1_right = w1_right
		self.w2_left = w2_left
		self.w2_leftright = w2_leftright
		self.w2_right = w2_right
		self.w3 = w3
	
	def forward_pass(self, x_left, x_right):
		b = np.mat(np.ones([1, x_left.shape[1]]), dtype=np.float) 

		# first layer
		a1_left = self.w1_left * x_left
		a1_right = self.w1_right * x_right

		z1_left = tanh(a1_left)
		z1_right = tanh(a1_right)
		z1_left_b = np.vstack([z1_left, b])
		z1_right_b = np.vstack([z1_right, b])

		# second layer
		a2_left = self.w2_left * z1_left_b
		a2_leftright = self.w2_leftright * np.vstack([z1_left, z1_right, b])
		a2_right = self.w2_right * z1_right_b

		z2 = m(a2_leftright, sigmoid(a2_left), sigmoid(a2_right))
		z2_b = np.vstack([z2, b])

		# third layer
		a3 = self.w3 * z2_b

		zs = z1_left_b,z1_right_b,z2_b
		ass = a1_left,a1_right,a2_left,a2_leftright,a2_right,a3
		return zs, ass

	def backward_pass(self, zs, ass, x_left, x_right, t):
		z1_left,z1_right,z2 = zs
		a1_left,a1_right,a2_left,a2_leftright,a2_right,a3 = ass

		# third layer
		if self.binary:
			r3 = sigmoid(a3) - np.float(0.5) * (t + np.float(1.0))
		else:
			#r3 = sigmoid(a3) - t
			r3 = a3 - t # we want to minimize squared error, so we use the simpler formula
		g3 = r3 * z2.T

		# second layer
		w3_r3 = self.w3[:,:-1].T * r3
		r2_left = m(a2_leftright, dxsigmoid(a2_left), sigmoid(a2_right), w3_r3)
		r2_leftright = m(sigmoid(a2_left), sigmoid(a2_right), w3_r3)
		r2_right = m(a2_leftright, sigmoid(a2_left), dxsigmoid(a2_right), w3_r3)

		g2_left = r2_left * z1_left.T
		g2_leftright = r2_leftright * np.vstack([z1_left[:-1,:], z1_right]).T
		g2_right = r2_right * z1_right.T

		# first layer
		dxtanh_a1_left = dxtanh(a1_left)
		r1_left_left = m(dxtanh_a1_left, self.w2_left[:,:-1].T * r2_left)
		r1_left_leftright = m(dxtanh_a1_left, self.w2_leftright[:,:self.H1].T * r2_leftright)
		r1_left = r1_left_left + r1_left_leftright

		dxtanh_a1_right = dxtanh(a1_right)
		r1_right_right = m(dxtanh_a1_right, self.w2_right[:,:-1].T * r2_right)
		r1_right_leftright = m(dxtanh_a1_right, self.w2_leftright[:,self.H1:-1].T * r2_leftright)
		r1_right = r1_right_right + r1_right_leftright

		g1_left = r1_left * x_left.T
		g1_right = r1_right * x_right.T

		return g1_left,g1_right,g2_left,g2_leftright,g2_right,g3

	def train(self, x_left, x_right, t):
		grads = self.gradients(x_left, x_right, t)
		ws = self.gradient.descend(grads, self.ws)
		self.update_ws(ws)

	def classify(self, x_left, x_right):
		b = np.mat(np.ones([1, np.mat(x_left).shape[1]])) 
		x_left = np.mat(np.vstack([np.mat(x_left), b]), dtype=np.float)
		x_right = np.mat(np.vstack([np.mat(x_right), b]), dtype=np.float)
		_, ass = self.forward_pass(x_left, x_right)
		if self.binary:
			result = np.sign(ass[-1])
		else:
			result = np.argmax(ass[-1], 0)
		return result

	def error(self, x_left, x_right, t):
		b = np.mat(np.ones([1, np.mat(x_left).shape[1]])) 
		x_left = np.mat(np.vstack([np.mat(x_left), b]), dtype=np.float)
		x_right = np.mat(np.vstack([np.mat(x_right), b]), dtype=np.float)
		_, ass = self.forward_pass(x_left, x_right)
		if self.binary:
			x = m(-t, ass[-1])
			neg = np.sum(np.log(1.0 + np.exp(x[x<0])), 1).flat[0]
			pos = np.sum(x[x>=0] + np.log(1.0 + np.exp(-x[x>=0])), 1).flat[0]
			error = neg + pos
			classerror = np.sum(np.sign(x[x>=0])) / t.shape[1]
		else:
			x = ass[-1] - t
			error = np.sum(0.5 * np.sum(m(x, x), 1), 0).flat[0]
			classerror = np.sum(np.argmax(ass[-1], 0) != np.argmax(t, 0), 0).flat[0]
		return error, classerror
