from numpy import *
from scipy.io import *


def load_data():
	data = loadmat('norb_binary.mat')
	train['l'] = data['train_left_s']
	train['r'] = data['train_right_s']
	train['t'] = data['train_cat_s']
	test['l']  = data['test_left_s']
	test['r']  = data['test_right_s']
	test['t']  = data['test_cat_s']
	return train, test


def split_data(dataset):
	pass


def preprocess(dataset):
	mu = mean(dataset,1)
	sigma = 1/std(dataset,0)
	normalized = sigma * (dataset - mu)	
	return normalized


def main():
	train, test = load_data()
	train_set, validation_set = split_data(train)
	
