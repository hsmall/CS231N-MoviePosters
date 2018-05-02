import numpy as np
import sklearn


class Dataset:
	def __init__(self, X, y):
		assert X.shape[0] == y.shape[0], 'Got different numbers of data and labels'
		self.X, self.y = X, y

	def size(self):
		return len(self.y)

	def image_size(self):
		return X.shape[1:]

	def get_mean_image(self):
		return np.mean(self.X, axis=0)

	def get_batches(self, batch_size):
		X_shuffled, y_shuffled = sklearn.utils.shuffle(self.X, self.y)

		num_batches = int(np.ceil(self.size()/batch_size))
		for i in range(num_batches):
			X_batch = X_shuffled[batch_size*i:batch_size*(i+1)]
			y_batch = y_shuffled[batch_size*i:batch_size*(i+1)] 
			yield X_batch, y_batch


def main():
	X = np.random.normal(size = (10, 5, 5, 3))
	y = np.random.normal(size = (10, 2))
	dataset = Dataset(X, y)


if __name__ == '__main__':
	main()
