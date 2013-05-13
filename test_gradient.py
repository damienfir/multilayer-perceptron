import sys
import mlp
import stream_utils as streams
import numpy as np

# easily tunable params
inputs = 100
count = 100
h = 1e-7

stream = streams.training_binary()
classifier = mlp.MLP(10, 10, k=2)

class DirectionalGradientGenerator:

	def __init__(self, x_left, x_right, t):
		self.mlp_plus = classifier.clone()
		self.mlp_minus = classifier.clone()
		self.x_left = x_left
		self.x_right = x_right
		self.t = t
		self.random = np.random.RandomState(1234)

	def compute(self, h):
		idx = int(self.random.rand() * 6.0)
		w_plus,w_minus = self.mlp_plus.ws[idx],self.mlp_minus.ws[idx]
		x,y = int(self.random.rand() * float(w_plus.shape[0])), int(self.random.rand() * float(w_plus.shape[1]))

		# pertubate weight vectors
		w_plus[x,y] = w_plus[x,y] + h
		w_minus[x,y] = w_minus[x,y] - h

		# compute gradient
		r_plus, _ = self.mlp_plus.error(self.x_left, self.x_right, self.t)
		r_minus, _ = self.mlp_minus.error(self.x_left, self.x_right, self.t)

		# reset weight vectors for next time
		w_plus[x,y] = w_plus[x,y] - h
		w_minus[x,y] = w_minus[x,y] + h
		return idx, x, y, (r_plus - r_minus) / (2.0 * h)

sys.stdout.write("Verifying gradient computation ... ")
sys.stdout.flush()

errors = []
for _ in xrange(0, inputs):
	x_left, x_right, t = stream.next(count=10)
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
