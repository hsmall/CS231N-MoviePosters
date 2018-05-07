import logging
import os
# pip3 install https://pypi.python.org/packages/source/P/PrettyTable/prettytable-0.7.2.tar.bz2
#from prettytable import PrettyTable
#import progressbar  # pip install progressbar2
import sklearn.metrics
import tensorflow as tf
import numpy as np 

class Model:
	def __init__(self, genres):
		print('Initializing model...', end='')
		self.genres = genres
		self.num_classes = len(genres)
		self.session = tf.Session()

		self.output = self.build_graph()
		self.loss = self.loss_fn(self.y_placeholder, self.output)
		self.train_op = self.optimizer(self.loss)
		
		self.session.run(tf.global_variables_initializer())
		print('Done.')

	def build_graph(self):
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

	def train(self, train_dataset, valid_dataset, num_epochs=10, batch_size=1000, learning_rate=1e-3, verbose=False):
		print('Training Model...')
		best_valid_loss = float('inf')
		for epoch in range(1, num_epochs+1):
			
			# Initialize the progress bar
			num_batches = int(np.ceil(train_dataset.size()/batch_size))
			'''
			progbar = progressbar.ProgressBar(
				widgets = [
					'Epoch #{0} out of {1}: '.format(epoch, num_epochs),
					' ', progressbar.Percentage(),
					' | ', progressbar.DynamicMessage('loss'),
					' | ', progressbar.DynamicMessage('accuracy'),
					' | ', progressbar.ETA(),
				],
				max_value = num_batches
			)
			'''

			# Train on all batches for this epoch
			for batch, (X_batch, y_batch) in enumerate(train_dataset.get_batches(batch_size)):
				train_loss, _ = self.session.run((self.loss, self.train_op), {
					self.X_placeholder : X_batch,
					self.y_placeholder : y_batch,
					self.learning_rate : learning_rate
				})
				train_accuracy = np.mean(self.get_accuracies(X_batch, y_batch))
				#progbar.update(batch+1, loss=train_loss, accuracy=train_accuracy)

			valid_loss = self.session.run(self.loss, {
				self.X_placeholder : valid_dataset.X,
				self.y_placeholder : valid_dataset.y,
			})

			marker = ""
			if valid_loss <= best_valid_loss:
				best_valid_loss = valid_loss
				self.save("saved_models/{0}/{0}".format(type(self).__name__))
				marker = "*"

			print('Validation Loss: {0:.4f} {1}'.format(valid_loss, marker))
			#if verbose:
			#	print(self.get_stats_table(valid_dataset.X, valid_dataset.y))
		print('Done Training.')
			

	def predict(self, dataset):
		raise NotImplementedError('Must implement predict().')

	def get_accuracies(self, X, y):
		preds = self.predict(X)
		return [ np.mean(y[:, i] == preds[:, i]) for i in range(self.num_classes)]

	def get_precision_recall_f1(self, X, y):
		preds = self.predict(X)
		scores = []
		for i in range(self.num_classes):
			scores.append((
				sklearn.metrics.precision_score(y[:, i], preds[:, i]),
				sklearn.metrics.recall_score(y[:, i], preds[:, i]),
				sklearn.metrics.f1_score(y[:, i], preds[:, i]),
			))
		return scores

	def get_stats_table(self, X, y):
		table = PrettyTable(['Genre', 'Accuracy', 'Precision', 'Recall', 'F1'])
		table.align['Genre'] = 'l'
		table.float_format = '.4'

		accuracies = self.get_accuracies(X, y)
		scores = self.get_precision_recall_f1(X, y)

		for i in range(self.num_classes):
			table.add_row([self.genres[i], accuracies[i]] + list(scores[i]))
		
		return table
	

	'''
	# Computes AUROC scores 
	def compute_auroc_scores(self, x_data, y_data):
		preds = self.predict(x_data)
		y = np.array(y_data)
		return [ roc_auc_score(y[:, i], preds[:, i]) for i in range(y.shape[1]) ]
	
	def compute_accuracies(self, x_data, y_data):
		correct_labels = np.equal(np.round(self.predict(x_data)), y_data)
		return np.mean(correct_labels, axis=0)
	'''
	


