import numpy as np
from utils import *

class LeastSquares:

	def __init__(self, v, k=5):
		self.v = v
		self.k = k

	def train(self, x_lefts, x_rights, ts):
		X = np.mat(np.vstack([x_lefts, x_rights, np.ones(x_lefts.shape[1])]), dtype=np.float).T
		A = X.T * X + self.v * np.diagflat(np.ones(X.shape[1]))
		b = X.T * ts.T

		# use numpy to solve the least squares normal equations problem
		ws, _, _, _ = np.linalg.lstsq(A, b)
		self.ws = ws

	def classify(self, x_left, x_right):
		X = np.mat(np.vstack([x_left, x_right, np.ones(x_left.shape[1])]), dtype=np.float)
		return np.argmax(X.T * self.ws, 1).T

	def error(self, x_left, x_right, t):
		X = np.mat(np.vstack([x_left, x_right, np.ones(x_left.shape[1])]), dtype=np.float)
		result = (X.T * self.ws).T
		x = result - t
		error = np.sum(0.5 * np.sum(m(x, x), 1), 0).flat[0]
		classerror = np.sum(np.argmax(result, 0) != np.argmax(t, 0), 0).flat[0]
		return error, classerror
