from mlp import MLP

mlp = MLP(None, 5, 5, 576)

data = scipy.io.loadmat('norb/processed_binary.mat')

x_left = np.matrix(data['training_left'][:,0])
x_right = np.matrix(data['training_right'][:,0])
t = np.matrix(data['training_cat'][:,0])

# gradient from backpropagation
mlp.train(x_left,x_right,t)
g_backprop = mlp.grads


# values from forward passes
h = 1e-2
mlp.w_add(h)
mlp.train(x_left,x_right,t)
hplus = mlp.ass

mlp.w_add(-h)
mlp.train(x_left,x_right,t)
hminus = mlp.ass


# difference between two gradients
err = g_backprop - (hplus - hminus)/(2*h)

print err
