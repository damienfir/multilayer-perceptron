import numpy as np
from utils import *
import gradient

class LogisticLoss:

	def __init__(self, nu, mu, dimension=576, k=5, seed=12345):
		self.gradient = gradient.Gradient(nu, mu)
		self.k = 5
		random = np.random.RandomState(seed)
		self.ws = np.mat(random.randn(k, 2 * dimension + 1), dtype=np.float)

	def train(self, x_lefts, x_rights, ts):
		X = np.mat(np.vstack([x_lefts, x_rights, np.ones(x_lefts.shape[1])]), dtype=np.float)
		t = np.mat(np.vstack(map(lambda k: ts == k, range(1, self.k + 1))), dtype=np.float)
		grads = (sigmoid(self.ws * X) - t) * X.T
		print grads.shape, self.ws.shape
		self.ws = self.gradient.descend(grads, self.ws)

	def classify(self, x_left, x_right):
		X = np.mat(np.vstack([x_left, x_right, np.ones(x_left.shape[1])]), dtype=np.float)
		result = np.argmax(self.ws * X, 0)
		return result + 1

	def error(self, x_left, x_right, t):
		X = np.mat(np.vstack([x_left, x_right, np.ones(x_left.shape[1])]), dtype=np.float)
		raise Exception("TODO")
		result = np.argmax(self.ws * X, 0)
		return result + 1
