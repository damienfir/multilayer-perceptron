import mlp
import scipy.io
from gradient import Gradient
import numpy as np

data = scipy.io.loadmat('norb/processed_binary.mat')
x_left = np.matrix(data['training_left'][:,0])
x_right = np.matrix(data['training_right'][:,0])
t = np.matrix(data['training_cat'][:,0])
print x_left.shape, x_right.shape, t.shape
gradient = Gradient(0.01,0.1)
mlp = mlp.MLP(gradient, 5, 5, 576)
mlp.train(x_left,x_right,t)
