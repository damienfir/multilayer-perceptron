import numpy as np
from utils import *
import gradient

class LogisticLoss:

	def __init__(self, nu, mu, dimension=576, k=5, seed=123456):
		self.gradient = gradient.Gradient(nu, mu)
		self.k = 5
		self.dimension = dimension
		random = np.random.RandomState(seed)
		self.ws = np.mat(random.randn(2 * dimension + 1, k), dtype=np.float)

	def clone(self):
		classifier = LogisticLoss(self.gradient.nu, self.gradient.mu,
			dimension=self.dimension, k=self.k)
		classifier.ws = np.copy(self.ws)
		return classifier

	def gradients(self, x_left, x_right, t):
		b = np.mat(np.ones(x_left.shape[1]), dtype=np.float)
		X = np.mat(np.vstack([x_left, x_right, b]), dtype=np.float)
		y = self.ws.T * X
		sigma_k = np.exp(y - lsexp(y))
		grad_Ei = sigma_k - t
		grads = (grad_Ei * X.T).T
		return grads

	def train(self, x_left, x_right, t):
		grads = self.gradients(x_left, x_right, t)
		self.ws = self.gradient.descend(grads, self.ws)

	def classify(self, x_left, x_right):
		b = np.mat(np.ones(x_left.shape[1]), dtype=np.float)
		X = np.mat(np.vstack([x_left, x_right, b]), dtype=np.float)
		result = np.argmax(self.ws * X, 0)
		return result

	def error(self, x_left, x_right, t):
		b = np.mat(np.ones(x_left.shape[1]), dtype=np.float)
		X = np.mat(np.vstack([x_left, x_right, b]), dtype=np.float)
		y = self.ws.T * X
		error = np.sum(lsexp(y, 0) - np.sum(m(t, y), 0), 1).flat[0]
		classerror = np.sum(np.argmax(y, 0) != np.argmax(t, 0), 1).flat[0]
		return error, classerror

	def normalized_error(self, *args):
		if len(args) == 1:
			x_left, x_right, t = args[0].all()
			size = float(args[0].size)
		elif len(args) == 3:
			x_left, x_right, t = args
			size = float(x_left.shape[1])
		else: raise Exception("Can't run error on weird args " + str(args))
		error, classerror = self.error(x_left, x_right, t)
		normalized_error = float(error) / size
		normalized_classerror = float(classerror) / size
		return normalized_error, normalized_classerror
