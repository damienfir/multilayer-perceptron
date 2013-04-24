import numpy as np
from utils import *
import gradient

class LogisticLoss:

	def __init__(self, nu, mu, dimension=576, k=5, seed=12345):
		self.gradient = gradient.Gradient(nu, mu)
		self.k = 5
		random = np.random.RandomState(seed)
		self.ws = np.mat(random.randn(1, 2 * dimension + 1), dtype=np.float)

	def train(self, x_lefts, x_rights, ts):
		x_lefts = np.mat(x_lefts, dtype=np.float)
		x_rights = np.mat(x_rights, dtype=np.float)
		X = np.mat(np.vstack([x_lefts, x_rights, np.ones(x_lefts.shape[1])]), dtype=np.float)
		t = np.mat(np.vstack(map(lambda k: ts == k, range(1, self.k + 1))), dtype=np.float)
		grads = sigmoid(self.ws * X) - t
		print grads.shape, self.ws.shape
		self.ws = self.gradient.descend(grads, self.ws)

	def classify(self, x_left, x_right):
		x_left = np.mat(x_left, dtype=np.float)
		x_right = np.mat(x_right, dtype=np.float)
		X = np.mat(np.vstack([x_left, x_right, np.ones(x_left.shape[1])]), dtype=np.float)
		result = np.argmax(X.T * self.ws, 1)
		return (result + 1).T
