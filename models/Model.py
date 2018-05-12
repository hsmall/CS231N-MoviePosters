import logging
import os
# pip3 install https://pypi.python.org/packages/source/P/PrettyTable/prettytable-0.7.2.tar.bz2
#from prettytable import PrettyTable
from models.util import Progbar
import sklearn.metrics
import tensorflow as tf
import numpy as np 

class Model:
	def __init__(self, genres):
		print('Initializing model...', end='')
		self.genres = genres
		self.num_classes = len(genres)
		self.session = tf.Session()
		self.is_training = tf.placeholder(tf.bool, shape=())

		self.output = self.build_graph()
		self.loss = self.loss_fn(self.y_placeholder, self.output)
		self.train_op = self.optimizer(self.loss)
		
		self.session.run(tf.global_variables_initializer())
		print('Done.')

	def build_graph(self):
		# Must initialize self.X_placeholder, self.y_placeholder and
		# self.learning_rate. Must return tensor that represents model
		# output (used in loss function).
		raise NotImplementedError('Must implement build_graph().')

	def loss_fn(self, y, output):
		raise NotImplementedError('Must implement loss_fn().')

	def optimizer(self):
		raise NotImplementedError('Must implement optimizer().')

	def save(self, filename):
		directory = os.path.dirname(filename)
		if not os.path.exists(directory):
			os.makedirs(directory)
		
		tf.train.Saver().save(self.session, filename)

	def load(self, filename):
		tf.train.Saver().restore(self.session, filename)

	def train(self, train_dataset, valid_dataset, num_epochs=10, batch_size=1000, eta=1e-3, saving=False, verbose=False):
		print('Training Model...')
		best_valid_loss = float('inf')
		for epoch in range(1, num_epochs+1):
			if epoch > 1: eta /= 2
			# Initialize the progress bar
			num_batches = int(np.ceil(train_dataset.size()/batch_size))     
			progbar = Progbar(target = num_batches)
            
			# Train on all batches for this epoch
			print('Epoch #{0} out of {1}: '.format(epoch, num_epochs))
			for batch, (X_batch, y_batch) in enumerate(train_dataset.get_batches(batch_size)):
				train_loss, _ = self.session.run((self.loss, self.train_op), {
					self.X_placeholder : X_batch,
					self.y_placeholder : y_batch,
					self.learning_rate : eta,
					self.is_training : True
				})
				progbar.update(batch+1, [('Train Loss', train_loss)])

			valid_loss = self.get_loss(valid_dataset, batch_size)
			marker = ""
			if valid_loss <= best_valid_loss:
				best_valid_loss = valid_loss
				self.save("saved_models/{0}/{0}".format(type(self).__name__))
				marker = "*"

			print('Validation Loss: {0:.4f} {1}'.format(valid_loss, marker))
			if verbose:
				print(self.get_stats_table(valid_dataset, batch_size))
		print('Done Training.')
			

	def predict(self, X, batch_size=None):
		raise NotImplementedError('Must implement predict().')

	def get_loss(self, dataset, batch_size=None):
		if batch_size is None: batch_size = len(X)

		losses = []
		X, y = dataset.X, dataset.y
		for i in range(0, len(X), batch_size):
			X_batch, y_batch = X[i:i+batch_size], y[i:i+batch_size]
			loss = self.session.run(self.loss, {
				self.X_placeholder : X_batch,
				self.y_placeholder : y_batch,
				self.is_training : False
			})
			losses.append(loss * len(X_batch))
		return np.sum(losses) / len(X)

                               
	def get_accuracies(self, predictions, y):
		return [ np.mean(y[:, i] == predictions[:, i]) for i in range(self.num_classes)]    

	def get_precision_recall_f1(self, predictions, y):
		scores = []
		for i in range(self.num_classes):
			scores.append((
				sklearn.metrics.precision_score(y[:, i], predictions[:, i]),
				sklearn.metrics.recall_score(y[:, i], predictions[:, i]),
				sklearn.metrics.f1_score(y[:, i], predictions[:, i]),
			))
		return scores

	def get_stats_table(self, dataset, batch_size):
		max_genre_len = max([len(genre) for genre in self.genres])     
		table  = '------------------------------------------------------------\n'
		table += '| {0:^{1}} | Accuracy | Precision | Recall |   F1   |\n'.format('Genre', max_genre_len)
		table += '------------------------------------------------------------\n'  
        
		predictions = self.predict(dataset.X, batch_size)
		accuracies = self.get_accuracies(predictions, dataset.y)
		scores = self.get_precision_recall_f1(predictions, dataset.y)
        
		row_format = '| {0:<{5}} |  {1:.4f}  |   {2:.4f}  | {3:.4f} | {4:.4f} |\n'                       
		for i in range(self.num_classes):
			row = [self.genres[i], accuracies[i]] + list(scores[i])
			table += row_format.format(*(row + [max_genre_len]))   

		final_row = ['Overall'] + [np.mean(accuracies)] + list(np.mean(scores, axis=0))
		table += '------------------------------------------------------------\n'
		table += row_format.format(*(final_row + [max_genre_len]))                     
		table += '------------------------------------------------------------\n'
		return table


