import numpy as np

class Gradient:

	def __init__(self, nu, mu):
		self.nu = nu
		self.mu = mu
		self.previous = None

	def descend(self, gradients, ws):
		if not (type(gradients) is list or type(gradients) is tuple):
			gradients = [gradients]
			ws = [ws]
			single = True
		else:
			single = False

		if self.previous == None and len(gradients) > 0:
			self.previous = []
			for m in gradients:
				self.previous.append(np.mat(np.zeros(m.shape)))
		if len(gradients) != len(ws) or len(self.previous) != len(ws):
			raise Exception("incompatible inputs: shapes don't match")

		def update_wk(args):
			gradient,previous,w = args
			delta_wk = -self.nu * (1 - self.mu) * gradient + self.mu * previous
			return (w + delta_wk, delta_wk)
		new = map(update_wk, zip(gradients, self.previous, ws))
		result,previous = map(lambda x: x[0], new), map(lambda x: x[1], new)
		self.previous = previous
		return result[0] if single else result
