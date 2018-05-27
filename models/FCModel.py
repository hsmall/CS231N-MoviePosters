from models.Model import Model

import numpy as np
import os
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

class FCModel(Model):
	def __init__(self, genres, label_probs, image_shape, hidden_layer_sizes):
		self.label_probs = label_probs
		self.image_shape = image_shape
		self.hidden_layer_sizes = hidden_layer_sizes
		Model.__init__(self, genres)

	def build_graph(self):
		self.X_placeholder = tf.placeholder(tf.float32, [None] + list(self.image_shape))
		self.y_placeholder = tf.placeholder(tf.float32, [None, self.num_classes])
		self.learning_rate = tf.placeholder(tf.float32, shape=())
		self.reg = 0.0#10000.0
		self.dropout = 0.5#0.5
       
		layers = [tf.layers.dropout(
			inputs = tf.contrib.layers.flatten(self.X_placeholder),
			rate = self.dropout,
			training = self.is_training,
		)]
        
		for hidden_layer_size in self.hidden_layer_sizes:
			layers.append(tf.layers.dropout(
				inputs = tf.layers.dense(
					inputs = layers[-1],
					units = hidden_layer_size,
					kernel_initializer = tf.contrib.layers.xavier_initializer(),
					bias_initializer = tf.zeros_initializer(),
					kernel_regularizer = tf.contrib.layers.l2_regularizer(self.reg),
					activation = tf.nn.tanh
				),
				rate = self.dropout,
				training = self.is_training,
			))
        
		output = tf.layers.dense(
			inputs = layers[-1],
			units = self.num_classes,
			kernel_initializer = tf.contrib.layers.xavier_initializer(),
			bias_initializer = tf.zeros_initializer(),
			kernel_regularizer = tf.contrib.layers.l2_regularizer(self.reg),
			activation = tf.nn.sigmoid
		)
        
		return output
    
	def loss_fn(self, y, output):
		pos_cross_entropy = y / self.label_probs * tf.log(tf.clip_by_value(output, 1e-10, 1.0))
		neg_cross_entropy = (1-y) / (1-self.label_probs) * tf.log(tf.clip_by_value(1-output, 1e-10, 1.0))
		return -tf.reduce_mean(pos_cross_entropy + neg_cross_entropy)
        
	def optimizer(self, loss):
		return tf.train.AdamOptimizer(self.learning_rate).minimize(loss)

	def predict(self, X, batch_size=None):
		if batch_size is None: batch_size = len(X)

		preds = np.zeros((len(X), self.num_classes))
		for i in range(0, len(X), batch_size): 
			preds[i:i+batch_size] = np.round(self.session.run(self.output, {
				self.X_placeholder : X[i:i+batch_size],
				self.is_training : False
			}))
		return preds

