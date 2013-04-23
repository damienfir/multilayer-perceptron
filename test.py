import mlp
import scipy.io
import numpy as np

data = scipy.io.loadmat('norb/processed_binary.mat')
classifier = mlp.MLP(0.01, 0.1, 5, 5, 576)
for i in xrange(0, 10):
	x_left = data['training_left'][:,i]
	x_right = data['training_right'][:,i]
	t = data['training_cat'][0,i]
	error = classifier.verify(x_left, x_right, t)
	print error
#	classifier.train(x_left,x_right,t)
