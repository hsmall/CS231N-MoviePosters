from PIL import Image
import scipy.misc
import glob
import numpy as np
import sys

class GenreDataset():
        def __init__(self, genre, batch_size):
                dirname = "posters/"+genre
                image_list = []
                for filename in glob.glob('posters/' + genre + '/*.jpg'):
                    image_list.append(filename)
                        # image_data = Image.open(filename)
                        # image_data = image_data.resize((64, 64))
                        # img_arr = np.array(image_data)
                        # # print("Original", img_arr)
                        # img_arr = np.divide(img_arr, 127.5)
                        # # print("Divided", img_arr)
                        # img_arr -= 1
                        # # print("Subtracted", img_arr)
                        # # sys.exit()
                        # image_list.append(img_arr)
                self.images = image_list
                print(len(self.images))
                # self.X = np.array(image_list, dtype='float32')
                # # self.X = self.X.astype(np.float32)/255
                # print(self.X.shape)
                # self.X = self.X[:, :, :, :3]
                self.batch_size = batch_size
                # print(self.X.shape) 
        def get_batch(self, idx):
            batch_files = self.images[idx*self.batch_size:idx*self.batch_size+self.batch_size] 
            batch = []
            for filename in batch_files: 
                # img = Image.open(filename) 
                # img = img.resize((64, 64))
                # img = np.array(img)/225.0
                # img = img.astype(np.float32)
                # img /= 2.0
                img = scipy.misc.imread(filename).astype(np.float) 
                img = scipy.misc.imresize(img, (64, 64))
                img = np.array(img)/255 
                img *= 2.0
                img -= 1.0
                # img = np.divide(img, 255)
                # img -= 1 
                batch.append(img) 
            batch = np.array(batch).astype(np.float32) 
            # batch = np.reshape(batch, (batch.shape[0], -1))
            return batch

        def num_batches(self):
                print(np.ceil(len(self.images)/self.batch_size)) 
                return int(np.ceil(len(self.images)/self.batch_size))

        def __iter__(self): 
                N, B = self.X.shape[0], self.batch_size 
                idxs = np.arange(N)

                return iter((self.X[i:i+B]) for i in range(0, N, B))

if __name__ == '__main__':
        main()
