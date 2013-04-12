import random
import sys, os
import scipy.io, numpy as np

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
	mean = array.sum(1) / n
	mean_matrix = np.tile(mean, (n, 1)).transpose()
	mean_difference = array - mean_matrix
	squared_difference = mean_difference ** 2
	squared_difference_sum = squared_difference.sum(1)
	squared_difference_mean = squared_difference_sum / n
	sigma = squared_difference_mean ** 0.5
	sigma_minus = 1.0 / sigma
	sigma_minus_matrix = np.tile(sigma_minus, (n, 1)).transpose()
	normalized_matrix = sigma_minus_matrix * mean_difference
	return normalized_matrix

norb_normalized_left = normalize(norb_training_left)
norb_normalized_right = normalize(norb_training_right)

# split data into training and validation sets
training_count = int(2.0 * norb_train_cat.size / 3.0)
indices = range(norb_train_cat.size)
random.shuffle(indices) # shuffle in place
training_indices = indices[:training_count]
validation_indices = indices[training_count:]

final_training_left = norb_normalized_left[:,training_indices]
final_training_right = norb_normalized_right[:,training_indices]
final_training_cat = norb_train_cat[:,training_indices]

final_validation_left = norb_normalized_left[:,validation_indices]
final_validation_right = norb_normalized_right[:,validation_indices]
final_validation_cat = norb_train_cat[:,validation_indices]

# write normalized and split data back to file for future use
data = {
	'training_left': final_training_left,
	'training_right': final_training_right,
	'training_cat': final_training_cat,
	'validation_left': final_validation_left,
	'validation_right': final_validation_right,
	'validation_cat': final_validation_cat
}	

basename = os.path.basename(path)
dirname = os.path.dirname(path)
qualification = basename.split('_', 1)[1]
save_path = dirname + '/processed_' + qualification
scipy.io.savemat(save_path, data)
