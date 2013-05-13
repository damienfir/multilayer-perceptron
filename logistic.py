import numpy as np
from utils import *
import gradient

class LogisticLoss:

	def __init__(self, nu, mu, dimension=576, k=5, seed=12345):
		self.gradient = gradient.Gradient(nu, mu)
		self.k = 5
		self.dimension = dimension
		random = np.random.RandomState(seed)
		self.ws = np.mat(random.randn(k, 2 * dimension + 1), dtype=np.float)

	def clone(self):
		classifier = LogisticLoss(self.gradient.nu, self.gradient.mu,
			dimension=self.dimension, k=self.k)
		classifier.ws = self.ws
		return classifier

	def train(self, x_left, x_right, t):
		X = np.mat(np.vstack([x_left, x_right, np.ones(x_left.shape[1])]), dtype=np.float)
		grads = (sigmoid(self.ws * X) - t) * X.T
		self.ws = self.gradient.descend(grads, self.ws)

	def classify(self, x_left, x_right):
		X = np.mat(np.vstack([x_left, x_right, np.ones(x_left.shape[1])]), dtype=np.float)
		result = np.argmax(self.ws * X, 0)
		return result

	def error(self, x_left, x_right, t):
		X = np.mat(np.vstack([x_left, x_right, np.ones(x_left.shape[1])]), dtype=np.float)
		result = self.ws * X
		x = m(-t, result)
		neg = np.sum(np.log(1.0 + np.exp(x[x<0])), 1).flat[0]
		pos = np.sum(x[x>=0] + np.log(1.0 + np.exp(-x[x>=0])), 1).flat[0]
		error = neg + pos
		classerror = np.sum(np.argmax(result, 0) != np.argmax(t, 0), 1).flat[0]
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
