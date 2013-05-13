import scipy.io
import numpy as np
import random

class Stream:

	def __init__(self, data, direction='horizontal', indices=None):
		if type(data) is tuple:
			self.data = data
		elif type(data) is list:
			self.data = tuple(data)
		else:
			self.data = (data,)
		if direction == 'horizontal':
			self.direction = 'h'
		elif direction == 'vertical':
			self.direction = 'v'
		else:
			raise Exception("unknown direction " + str(direction))
		if indices != None:
			self.indices = indices
		else:
			if self.direction == 'h':
				size = self.data[0].shape[1]
			else:
				size = self.data[0].shape[0]
			self.indices = range(size)
			random.shuffle(self.indices)
		self.read = 0

	def __getitem__(self, idx):
		if not type(idx) is int:
			raise Exception("unkown index type: " + str(type(idx)))
		idx = self.indices[idx]
		result = []
		for col in self.data:
			read = col[:,idx:idx+1] if self.direction == 'h' else col[idx:idx+1,:]
			result.append(read)
		return tuple(result)

	def next(self,count=1):
		if self.read + count > self.size:
			first = self.indices[self.read:]
			random.shuffle(self.indices)
			self.read = (self.read + count) - self.size
			last = self.indices[:self.read]
			indices = first + last
			self.looped = True
		else:
			indices = self.indices[self.read:self.read + count]
			self.read = self.read + count
			self.looped = False
		result = []
		for col in self.data:
			read = col[:,indices] if self.direction == 'h' else col[indices,:]
			result.append(read)
		return tuple(result)

	def all(self):
		result = []
		for col in self.data:
			read = col[:,self.indices] if self.direction == 'h' else col[self.indices,:]
			result.append(read)
		return tuple(result)

	@property
	def size(self):
		return len(self.indices)
