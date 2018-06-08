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
                self.images = image_list
                print(len(self.images))
                self.batch_size = batch_size
                print('batch_size', batch_size)

        def crop_image(self, img, width, height):
            h, w, c = img.shape 
            j = int(np.ceil((h-width)/2.0))
            i = int(np.ceil((w-height)/2.0))
            cropped = img[j:j+width, i:i+height]
            return cropped

        def get_batch(self, idx):
            batch_files = self.images[idx*self.batch_size:idx*self.batch_size+self.batch_size] 
            batch = []
            for filename in batch_files: 
                img = Image.open(filename)
                img = img.convert('RGB')
                img = img.resize((64, 64), Image.ANTIALIAS)
                img = np.array(img).astype(np.float32)
                h, w, c = img.shape 
                img /= 255.0 
                img *= 2.0
                img -= 1.0
                batch.append(img) 
            batch = np.array(batch).astype(np.float32) 
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
