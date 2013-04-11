from numpy import *

class MLP:
	def train(self):
		pass
	
	def backwards(self):
		pass

	def forward(self, x_l, x_r):
		# first layer
		a1_l = self.w1_l.T * x_l + self.b1_l
		z1_l = tanh(a1_l)
		a1_r = self.w1_r.T * x_r + self.b1_r
		z1_r = tanh(a1_r)
	
		# second layer
		a2_l = self.w2_l.T * z1_l + self.b2_l
		a2_r = self.w2_r.T * z1_r + self.b2_r
		a2_rl = self.w2_rl.T * hstack(z1_l.T,z1_r.T) + self.b2_rl
		z2 = a2_rl * sigmoid(a2_l) * sigmoid(a2_r)

		# third layer
		a3 = self.w3.T * z2 + self.b3

		return a3


def sigmoid(x):
	return (1 + exp(-x))
