from utils import *
import gradient
import scipy.io
import numpy as np

class MLP:

	def __init__(self, nu, mu, H1, H2, dimension=576, binary=True, seed=1234567890, copy=None):
		self.gradient = gradient.Gradient(nu, mu)
		self.H1 = H1
		self.H2 = H2
		self.dim = dimension
		self.binary = binary
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
			self.w3 = rand(1, H2)
		else:
			self.w1_left = np.copy(copy.w1_left)
			self.w1_right = np.copy(copy.w1_right)
			self.w2_left = np.copy(copy.w2_left)
			self.w2_leftright = np.copy(copy.w2_leftright)
			self.w2_right = np.copy(copy.w2_right)
			self.w3 = np.copy(copy.w3)

	def clone(self):
		return MLP(self.gradient.nu, self.gradient.mu, self.H1, self.H2, self.dim, copy=self)

	def gradients(self, x_left, x_right, t):
		b = np.mat(np.ones([1, np.mat(x_left).shape[1]])) 
		x_left = np.mat(np.vstack([np.mat(x_left), b]), dtype=np.float)
		x_right = np.mat(np.vstack([np.mat(x_right), b]), dtype=np.float)
		t = np.mat(t, dtype=np.float)
		t = np.mat(t if not self.binary else t - 2, dtype=np.float)
		zs, ass = self.forward_pass(x_left, x_right)
		grads = self.backward_pass(zs, ass, x_left, x_right, t)
		return grads

	def directional_gradients(self, _x_left, _x_right, _t):
		class DirectionalGradientGenerator:
			def __init__(self0, x_left, x_right, t):
				self0.mlp_plus = self.clone()
				self0.mlp_minus = self.clone()
				b = np.mat(np.ones([1, np.mat(x_left).shape[1]]))
				self0.x_left = np.mat(np.vstack([np.mat(x_left), b]), dtype=np.float)
				self0.x_right = np.mat(np.vstack([np.mat(x_right), b]), dtype=np.float)
				t = np.mat(t, dtype=np.float)
				self0.t = np.mat(t if not self.binary else t - 2, dtype=np.float)
				self0.random = np.random.RandomState(1234)

			def compute(self0, h):
				def logistic_error(res):
					error = np.sum(np.log(1.0 + np.exp(-self0.t * res[1][-1][0,0])), 1)
					if isinstance(error, (int, long, float, complex)): return error
					if isinstance(error[0], (int, long, float, complex)): return error[0]
					if isinstance(error[0,0], (int, long, float, complex)): return error[0,0]
					raise Exception("why am I here?? type of error is " + str(type(error)))
				# idx = int(self0.random.rand() * 6.0)
				idx = 4
				w_plus,w_minus = self0.mlp_plus.ws[idx],self0.mlp_minus.ws[idx]
				x,y = int(self0.random.rand() * float(w_plus.shape[0])), int(self0.random.rand() * float(w_plus.shape[1]))

				# pertubate weight vectors
				w_plus[x,y] = w_plus[x,y] + h
				w_minus[x,y] = w_minus[x,y] - h

				# compute gradient
				r_plus = logistic_error(self0.mlp_plus.forward_pass(self0.x_left, self0.x_right))
				r_minus = logistic_error(self0.mlp_minus.forward_pass(self0.x_left, self0.x_right))

				# reset weight vectors for next time
				w_plus[x,y] = w_plus[x,y] - h
				w_minus[x,y] = w_minus[x,y] + h
				return idx, x, y, (r_plus - r_minus) / (2.0 * h)

		return DirectionalGradientGenerator(_x_left, _x_right, _t)

	def verify(self, x_left, x_right, t, h=1e-8, count=100):
		grads = self.gradients(x_left, x_right, t)
		directionals = self.directional_gradients(x_left, x_right, t)
		errors = []
		for _ in xrange(0, count):
			idx,x,y,result = directionals.compute(h)
			mlp_result = grads[idx][x,y]
			if mlp_result: # check non zero for division
				ratio = abs(mlp_result - result) / abs(mlp_result)
				print ratio
				if ratio > h:
					errors.append((idx, x, y, mlp_result, result))
		return errors

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
		b = np.mat(np.ones([1, x_left.shape[1]])) 

		# first layer
		a1_left = self.w1_left * x_left
		a1_right = self.w1_right * x_right

		z1_left = tanh(a1_left)
		z1_right = tanh(a1_right)
		z1_left_b = np.vstack([z1_left, b])
		z1_right_b = np.vstack([z1_right, b])

		# second layer
		a2_left = self.w2_left * z1_left_b
		a2_right = self.w2_right * z1_right_b
		a2_leftright = self.w2_leftright * np.vstack([z1_left, z1_right, b])

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
		diagonalize = lambda mat: np.diagflat(np.sum(mat, 1))

		# third layer
		r3 = (sigmoid(a3) - 0.5 * (t + 1.0))

		g3 = r3 * z2.T

		# second layer
		r2_left = diagonalize(m(a2_leftright, dxsigmoid(a2_left), sigmoid(a2_right))) * self.w3[:,:-1].T * r3
		r2_right = diagonalize(m(a2_leftright, sigmoid(a2_left), dxsigmoid(a2_right))) * self.w3[:,:-1].T * r3
		r2_leftright = diagonalize(m(sigmoid(a2_left), sigmoid(a2_right))) * self.w3[:,:-1].T * r3

		g2_left = r2_left * z1_left.T
		g2_leftright = r2_leftright * np.vstack([z1_left[:-1,:], z1_right]).T
		g2_right = r2_right * z1_right.T

		# first layer
		a1_left_diagonal = diagonalize(dxtanh(a1_left))
		r1_left_left = a1_left_diagonal * self.w2_left[:,:-1].T * r2_left
		r1_left_leftright = a1_left_diagonal * self.w2_leftright[:,:self.H1].T * r2_leftright
		r1_left = r1_left_left + r1_left_leftright

		a1_right_diagonal = diagonalize(dxtanh(a1_right))
		r1_right_right = a1_right_diagonal * self.w2_right[:,:-1].T * r2_right
		r1_right_leftright = a1_right_diagonal * self.w2_leftright[:,self.H1:-1].T * r2_leftright
		r1_right = r1_right_right + r1_right_leftright

		g1_left = r1_left * x_left.T
		g1_right = r1_right * x_right.T

		return g1_left,g1_right,g2_left,g2_leftright,g2_right,g3

	def train(self, x_left, x_right, t):
		grads = self.gradients(x_left, x_right, t)
		ws = self.gradient.descend(grads, self.ws)
		self.update_ws(ws)
