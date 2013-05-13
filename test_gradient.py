import sys
import mlp, logistic
import stream_utils as streams
import numpy as np
import logistic

class DirectionalGradientGenerator:

	def __init__(self, classifier, x_left, x_right, t):
		self.classifier_pos = classifier.clone()
		self.classifier_neg = classifier.clone()
		example_ws = self.classifier_pos.ws
		self.single = not (type(example_ws) is list or type(example_ws) is tuple)
		self.x_left = x_left
		self.x_right = x_right
		self.t = t
		self.random = np.random.RandomState(1234)

	def compute(self, h):
		if not self.single:
			ws_pos, ws_neg = self.classifier_pos.ws, self.classifier_neg.ws
			idx = int(self.random.rand() * float(len(ws_pos)))
			w_pos, w_neg = ws_pos[idx], ws_neg[idx]
		else:
			idx = None
			w_pos, w_neg = self.classifier_pos.ws, self.classifier_neg.ws
		x,y = int(self.random.rand() * float(w_pos.shape[0])), int(self.random.rand() * float(w_pos.shape[1]))

		# pertubate weight vectors
		w_pos[x,y] = w_pos[x,y] + h
		w_neg[x,y] = w_neg[x,y] - h

		# compute gradient
		r_pos, _ = self.classifier_pos.error(self.x_left, self.x_right, self.t)
		r_neg, _ = self.classifier_neg.error(self.x_left, self.x_right, self.t)

		# reset weight vectors for next time
		w_pos[x,y] = w_pos[x,y] - h
		w_neg[x,y] = w_neg[x,y] + h
		return idx, x, y, (r_pos - r_neg) / (2.0 * h)

def verify(stream, classifier, count=100, inputs=100, h=1e-7):
	print " - Testing for (count, inputs, h)   :", count, inputs, h
	errors, skiped = [], 0
	for _ in xrange(0, inputs):
		x_left, x_right, t = stream.next(count=10)
		grads = classifier.gradients(x_left, x_right, t)
		directionals = DirectionalGradientGenerator(classifier, x_left, x_right, t)
		for _ in xrange(0, count):
			idx,x,y,directional_result = directionals.compute(h)
			if not directionals.single:
				classifier_result = grads[idx][x,y]
			else:
				classifier_result = grads[x,y]
			if abs(classifier_result) > 0.01:
				ratio = abs(classifier_result - directional_result) / abs(classifier_result)
				if ratio > h:
					errors.append((classifier_result, directional_result))
			else:
				skiped = skiped + 1
	
	error_count = len(errors)
	considered_count = (count * inputs) - skiped
	error_rate = 100.0 * float(error_count) / float(considered_count)
	summary = "%0.2f%% (%d / %d)" % (error_rate, error_count, considered_count)
	print "   result                           :", summary
	if errors:
		max_diff = max(map(lambda x: abs(x[0] - x[1]), errors))
		max_grad = max(map(lambda x: max(x[0], x[1]), errors))
		print "   -> maximal gradient              :", max_grad
		print "   -> maximal difference            :", max_diff
		#for idx, x, y, c_result, d_result in errors:
		#	print "ws[" + str(idx) + "][" + str(x) + "," + str(y) + "] ->",
		#	print "classifier gradient of", c_result, "vs directional of", d_result,
		#	print "diff =", abs(c_result - d_result)
	
def verify_binary_mlp():
	print "-- Binary MLP --------------------- :"
	stream = streams.training_binary()
	classifier = mlp.MLP(10, 10, k=2)
	verify(stream, classifier)

def verify_5class_mlp():
	print "-- 5Class MLP --------------------- :"
	stream = streams.training_5class()
	classifier = mlp.MLP(10, 10, k=5)
	verify(stream, classifier)

def verify_logistic():
	print "-- Logistic Regression ------------ :"
	stream = streams.training_5class()
	classifier = logistic.LogisticLoss(0.01, 0.1)
	verify(stream, classifier)

# verify_binary_mlp()
# verify_5class_mlp()
verify_logistic()
