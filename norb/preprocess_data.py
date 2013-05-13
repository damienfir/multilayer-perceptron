import random
import sys, os
import scipy.io, numpy as np

random.seed(123456)

if len(sys.argv) <= 1:
	print "Usage:",os.path.basename(sys.argv[0]),"path_to_matrix"
	sys.exit(1)
path = sys.argv[1]

# load the NORB data from the complete matrix
norb = scipy.io.loadmat(path)
norb_training_left = np.array(norb['train_left_s'])
norb_training_right = np.array(norb['train_right_s'])
norb_train_cat = np.array(norb['train_cat_s'])

# normalize the data as specified in the project description
def normalize(array):
	m,n = array.shape
	mean = array.sum(1) / float(n)
	mean_matrix = np.tile(mean, (n, 1)).transpose()
	mean_difference = array - mean_matrix
	squared_difference = mean_difference ** 2
	squared_difference_sum = squared_difference.sum(1)
	squared_difference_mean = squared_difference_sum / n
	sigma = squared_difference_mean ** 0.5
	sigma_minus = 1.0 / sigma
	sigma_minus_matrix = np.tile(sigma_minus, (n, 1)).transpose()
	normalized_matrix = sigma_minus_matrix * mean_difference
	return normalized_matrix, mean, sigma

norb_normalized_left, mean_left, sigma_left = normalize(norb_training_left)
norb_normalized_right, mean_right, sigma_right = normalize(norb_training_right)

# write normalized and split data back to file for future use
data = {
	'params_left': [mean_left, sigma_left],
	'params_right': [mean_right, sigma_right],
	'x_left': norb_normalized_left,
	'x_right': norb_normalized_right,
	't': norb_train_cat
}

basename = os.path.basename(path)
dirname = os.path.dirname(os.path.abspath(path))
qualification = basename.split('_', 1)[1]
save_path = dirname + '/processed_' + qualification
scipy.io.savemat(save_path, data)
