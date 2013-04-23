import mlp
import scipy.io
from gradient import Gradient
import numpy as np

data = scipy.io.loadmat('norb/processed_binary.mat')
x_left = data['training_left'][:,0]
x_right = data['training_right'][:,0]
t = data['training_cat'][0,0]
gradient = Gradient(0.01,0.1)
mlp = mlp.MLP(gradient, 5, 5, 576)
mlp.train(x_left,x_right,t)
