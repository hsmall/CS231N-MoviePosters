from PIL import Image
import glob
import numpy as np


class GenreDataset():
	def __init__(self, genre, batch_size):
		dirname = "posters/"+genre
		image_list = []
		for filename in glob.glob('posters/' + genre + '/*.jpg'):
			im=Image.open(filename)
			img=im.convert("RGB")
			im=img.resize((64, 64))
			image_list.append(np.array(im))

		self.X = np.array(image_list)
		self.X = self.X.astype(np.float32)/255
		self.X = self.X[:, :, :, :3]
		self.batch_size = batch_size
		print(self.X.shape) 

	def num_batches(self): 
		return int(np.ceil(self.X.shape[0])/self.batch_size)

	def __iter__(self): 
		N, B = self.X.shape[0], self.batch_size 
		idxs = np.arange(N)

		return iter((self.X[i:i+B]) for i in range(0, N, B))

if __name__ == '__main__':
	main()
