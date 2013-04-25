import scipy.io

class Stream:

	def __init__(self, path, keys, count=1, direction='horizontal'):
		self.data = scipy.io.loadmat(path)
		self.keys = keys
		self.count = count
		self.direction = direction
		if self.direction != 'horizontal' and self.direction != 'vertical':
			raise Exception("unknown direction " + str(self.direction))
		self.read = 0

	def __getitem__(self, idx):
		if not type(idx) is int:
			raise Exception("unkown index type: " + str(type(idx)))
		result = []
		for key in self.keys:
			if self.direction == 'horizontal':
				read = self.data[key][:,idx:idx+1]
			else:
				read = self.data[key][idx:idx+1,:]
			result.append(read)
		return tuple(result)

	def next(self):
		result = []
		for key in self.keys:
			if self.direction == 'horizontal':
				read = self.data[key][:,self.read:self.read+self.count]
			else:
				read = self.data[key][self.read:self.read+self.count,:]
			result.append(read)
		self.read += self.count
		return tuple(result)

	def all(self):
		result = []
		for key in self.keys:
			result.append(self.data[key][:,:])
		return tuple(result)

	@property
	def size(self):
		if self.direction == 'horizontal':
			size = self.data[self.keys[0]].shape[1]
		else:
			size = self.data[self.keys[0]].shape[0]
		return size
