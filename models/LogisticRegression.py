from models.Model import Model

import tensorflow as tf
import numpy as np 

class LogisticRegression(Model):
	def __init__(self, genres, input_dim):
		self.input_dim = input_dim
		Model.__init__(self, genres)

	def build_graph(self):
		self.X_placeholder = tf.placeholder(tf.float32, shape=[None, self.input_dim])
		self.y_placeholder = tf.placeholder(tf.float32, shape=[None, self.num_classes])
		self.learning_rate = tf.placeholder(tf.float32, shape=[])

		self.W = tf.Variable(tf.random_normal(shape=[self.input_dim, self.num_classes]))
		self.b = tf.Variable(tf.zeros(shape=[1, self.num_classes]))

		return tf.nn.sigmoid(tf.matmul(self.X_placeholder, self.W) + self.b)

	# Multiclass Cross-Entropy Loss
	def loss_fn(self, y, output):
		return -tf.reduce_mean(y*tf.log(tf.clip_by_value(output, 1e-10, 1.0)) + (1-y)*tf.log(tf.clip_by_value(1-output, 1e-10, 1.0)))

	def optimizer(self, loss):
		return tf.train.AdamOptimizer(self.learning_rate).minimize(loss)

	def predict(self, X):
		return self.session.run(self.output, {
			self.X_placeholder : X
		})

	def get_weights(self):
		return self.session.run(self.W)