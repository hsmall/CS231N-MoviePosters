from PIL import Image
import glob
import numpy as np
import sys

class GenreDataset():
        def __init__(self, genre, batch_size):
                dirname = "posters/"+genre
                image_list = []
                for filename in glob.glob('posters/' + genre + '/*.jpg'):
                        image_data = Image.open(filename)
                        image_data = image_data.resize((64, 64))
                        img_arr = np.array(image_data)
                        # print("Original", img_arr)
                        img_arr = np.divide(img_arr, 127.5)
                        # print("Divided", img_arr)
                        img_arr -= 1
                        # print("Subtracted", img_arr)
                        # sys.exit()
                        image_list.append(img_arr)

                self.X = np.array(image_list, dtype='float32')
                # self.X = self.X.astype(np.float32)/255
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
