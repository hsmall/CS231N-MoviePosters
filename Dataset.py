import numpy as np
import random

from keras.preprocessing.image import ImageDataGenerator

class Dataset:
    def __init__(self, X, y, mean=None, std=None, normalize=True):
        assert X.shape[0] == y.shape[0], 'Got different numbers of data and labels'
        assert (mean is None) == (std is None), 'Must specify both or neither of mean and std'

        self.mean = np.sum(X, axis=0) / len(X) if mean is None else mean
        #np.mean(X, axis=0) if mean is None else mean
        self.std = np.ones(X.shape[1:]) if std is None else std
        #np.std(X, axis=0) if std is None else std

        self.X = X.astype(np.float, copy=False)
        if normalize:
            self.X -= self.mean
            self.X /= self.std
        self.y = y

    def size(self):
        return len(self.y)

    def image_size(self):
        return X.shape[1:]

    def get_original_mean_and_std(self):
        return self.mean, self.std

    def get_batches(self, batch_size, augmentation=False):
        num_batches = int(np.ceil(self.size() / batch_size))
        
        if augmentation:
            data_generator = ImageDataGenerator(
                featurewise_center=False,
                samplewise_center=False,
                featurewise_std_normalization=False,
                samplewise_std_normalization=False,
                zca_whitening=False,
                zca_epsilon=1e-06,
                width_shift_range=10,
                height_shift_range=10,
                brightness_range=(0.5, 1.5),
                zoom_range=(0.5, 1.5),
                horizontal_flip=True,
            ).flow(self.X, self.y, batch_size)
            
            for i, (X_batch, y_batch) in enumerate(data_generator):
                if i >= num_batches: break
                yield X_batch, y_batch
        
        else:
            combined = list(zip(self.X, self.y))
            random.shuffle(combined)
            self.X[:], self.y[:] = zip(*combined)

            num_batches = int(np.ceil(self.size()/batch_size))
            for i in range(num_batches):
                X_batch = self.X[batch_size*i:batch_size*(i+1)]
                y_batch = self.y[batch_size*i:batch_size*(i+1)] 
                yield X_batch, y_batch

    @staticmethod
    def build_dataset_splits(X, y, splits=[0.75, 0.15, 0.10], normalize=False):
        assert len(splits) == 3, "splits must contain 3 numbers."
        assert np.sum(splits) == 1, "percentages in splits do not sum to 1."
        assert X.shape[0] == y.shape[0], 'Got different numbers of data and labels'

        num_examples = X.shape[0]
        train_split = int(np.ceil(num_examples * 0.75))
        valid_split = train_split + int(np.ceil(num_examples * 0.15))

        datasets = {}
        datasets['train'] = Dataset(
            X[:train_split],
            y[:train_split],
            normalize = normalize
        )
        train_mean, train_std = datasets['train'].get_original_mean_and_std()
   
        datasets['valid'] = Dataset(
            X[train_split:valid_split],
            y[train_split:valid_split],
            mean = train_mean,
            std = train_std,
            normalize = normalize
        )
        
        datasets['test'] = Dataset(
            X[valid_split:],
            y[valid_split:],
            mean = train_mean,
            std = train_std,
            normalize = normalize
        )

        return datasets


if __name__ == '__main__':
    X = np.random.normal(size = (10, 5, 5, 3))
    y = np.random.normal(size = (10, 2))
    dataset = Dataset(X, y)
