import sys
import mlp
import streamer
from utils import *
import numpy as np

# easily tunable params
inputs = 100
count = 100
h = 1e-7

path = 'norb/processed_binary.mat'
keys = ('training_left', 'training_right', 'training_cat')
stream = streamer.Stream(path, keys, count=10)
classifier = mlp.MLP(0.01, 0.1, 10, 10)

class DirectionalGradientGenerator:

	def __init__(self, x_left, x_right, t):
		self.mlp_plus = classifier.clone()
		self.mlp_minus = classifier.clone()
		b = np.mat(np.ones([1, np.mat(x_left).shape[1]]))
		self.x_left = np.mat(np.vstack([np.mat(x_left), b]), dtype=np.float)
		self.x_right = np.mat(np.vstack([np.mat(x_right), b]), dtype=np.float)
		t = np.mat(t, dtype=np.float)
		self.t = np.mat(t if not classifier.binary else t - 2, dtype=np.float)
		self.random = np.random.RandomState(1234)

	def compute(self, h):
		def logistic_error(res):
			x = m(- self.t, res[1][-1])
			neg = np.sum(np.log(1.0 + np.exp(x[x<0])), 1).flat[0]
			pos = np.sum(x[x>=0] + np.log(1.0 + np.exp(-x[x>=0])), 1).flat[0]
			return neg + pos
		#logistic_error = lambda res: np.sum(np.log(1.0 + np.exp(-self0.t * res[1][-1][0,0])), 1).flat[0]
		idx = int(self.random.rand() * 6.0)
		w_plus,w_minus = self.mlp_plus.ws[idx],self.mlp_minus.ws[idx]
		x,y = int(self.random.rand() * float(w_plus.shape[0])), int(self.random.rand() * float(w_plus.shape[1]))

		# pertubate weight vectors
		w_plus[x,y] = w_plus[x,y] + h
		w_minus[x,y] = w_minus[x,y] - h

		# compute gradient
		r_plus = logistic_error(self.mlp_plus.forward_pass(self.x_left, self.x_right))
		r_minus = logistic_error(self.mlp_minus.forward_pass(self.x_left, self.x_right))

		# reset weight vectors for next time
		w_plus[x,y] = w_plus[x,y] - h
		w_minus[x,y] = w_minus[x,y] + h
		return idx, x, y, (r_plus - r_minus) / (2.0 * h)

sys.stdout.write("Verifying gradient computation ... ")
sys.stdout.flush()

errors = []
for _ in xrange(0, inputs):
	x_left, x_right, t = stream.next()
	grads = classifier.gradients(x_left, x_right, t)
	directionals = DirectionalGradientGenerator(x_left, x_right, t)
	for _ in xrange(0, count):
		idx,x,y,result = directionals.compute(h)
		mlp_result = grads[idx][x,y]
		if mlp_result > 0.01 or abs(mlp_result - result) > 0.01:
			ratio = abs(mlp_result - result) / abs(mlp_result)
			if ratio > h:
				errors.append((idx, x, y, mlp_result, result))

print "[DONE]"
if errors:
	print "Errors found:"
	for idx, x, y, mlp_result, result in errors:
		print "ws[" + str(idx) + "][" + str(x) + "," + str(y) + "] ->",
		print "mlp gradient of", mlp_result, "vs directional of", result,
		print "diff =", abs(mlp_result - result)
	print "Total:", len(errors), "out of",
else:
	print "No errors uncovered after",
print count, "random directions in", inputs, "inputs =>", count * inputs, "tests"
