import numpy as np
import random
import sklearn


class Dataset:
	def __init__(self, X, y, mean=None, std=None):
		assert X.shape[0] == y.shape[0], 'Got different numbers of data and labels'
		assert (mean is None) == (std is None), 'Must specify both or neither of mean and std'

		self.mean = np.sum(X, axis=0) / len(X) if mean is None else mean
		#np.mean(X, axis=0) if mean is None else mean
		self.std = np.ones(X.shape[1:]) if std is None else std
		#np.std(X, axis=0) if std is None else std

		self.X = X.astype(np.float, copy=False)
		self.X -= self.mean
		self.X /= self.std
		self.y = y

	def size(self):
		return len(self.y)

	def image_size(self):
		return X.shape[1:]

	def get_original_mean_and_std(self):
		return self.mean, self.std

	def get_batches(self, batch_size):
		#X_shuffled, y_shuffled = sklearn.utils.shuffle(self.X, self.y)
        
		combined = list(zip(self.X, self.y))
		random.shuffle(combined)
		self.X[:], self.y[:] = zip(*combined)

		num_batches = int(np.ceil(self.size()/batch_size))
		for i in range(num_batches):
			X_batch = self.X[batch_size*i:batch_size*(i+1)]
			y_batch = self.y[batch_size*i:batch_size*(i+1)] 
			yield X_batch, y_batch

def main():
	X = np.random.normal(size = (10, 5, 5, 3))
	y = np.random.normal(size = (10, 2))
	dataset = Dataset(X, y)


if __name__ == '__main__':
	main()
