from numpy import *
from scipy.io import *

# load data
data = loadmat('norb_binary.mat')

# split data
train_l = data['train_left_s']
train_r = data['train_right_s']
train_cat = data['train_cat_s']

test_l = data['test_left_s']
test_r = data['test_right_s']
test_cat = data['test_cat_s']
