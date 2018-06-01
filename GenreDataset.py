from PIL import Image
import glob
import numpy as np


class GenreDataset():
	def __init__(self, genre, batch_size):
		dirname = "posters/"+genre
		image_list = []
		for filename in glob.glob('posters/' + genre + '/*.jpg'):
			print(filename)
			im=Image.open(filename)
			image_list.append(np.array(im))

		self.X = np.array(image_list)
		self.X = self.X.astype(np.float32)/255
		self.X = np.reshape(self.X, (self.X.shape[0], -1))
		self.batch_size = batch_size

	def __iter__(self): 
		N, B = self.X.shape[0], self.batch_size 
		idxs = np.arange(N)

		return iter((self.X[i:i+B], self.y[i:i+B]) for i in range(0, N))

if __name__ == '__main__':
	main()
