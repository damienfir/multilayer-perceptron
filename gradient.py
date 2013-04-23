
class Gradient:

	def __init__(self, nu, mu):
		self.nu = nu
		self.mu = mu
		self.previous = None

	def descend(self, gradients, ws):
		if self.previous == None:
			self.previous = (0,) * len(gradients)
		if len(gradients) != len(ws) or len(self.previous) != len(ws):
			raise Exception("incompatible inputs: shapes don't match")
		def update_wk(args):
			gradient,previous,w = args
			delta_wk = -nu * (1 - mu) * gradient + mu * previous
			return (w + delta_wk, delta_wk)
		new = map(update_wk, zip(gradients, self.previous, ws))
		result,previous = map(lambda x: x[0], new), map(lambda x: x[1], new)
		self.previous = previous
		return result